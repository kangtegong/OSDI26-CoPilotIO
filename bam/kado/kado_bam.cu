// kado_bam: the BaM GPU-direct I/O engine of kado (python module).
//
// One Device = one NVMe controller bound to the libnvm kernel module (/dev/libnvmN), opened
// through the BaM library (third_party/OSDI26-CoPilotIO/bam): admin queue on the host, I/O
// queue pairs in GPU memory. kado's GPU staging buffer is mapped for the SSD's DMA
// (nvidia_p2p through libnvm) once, at bind, and one PRP descriptor per I/O-sized piece of it
// is built then (prebuilt descriptors: nothing is built at submit time).
//
// A transfer is one CUDA kernel on the caller's stream. Each GPU thread issues ONE NVMe
// command (read or write of io_bytes) on queue pair (thread % n_qps), polls its completion
// and checks the NVMe status. The kernel's completion on the stream is the transfer's
// completion: a CUDA event recorded after it orders the consumers; no host thread polls.

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <ctrl.h>
#include <buffer.h>
#include <queue.h>
#include <nvm_cmd.h>
#include <nvm_dma.h>
#include <nvm_error.h>
#include <nvm_parallel_queue.h>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// desc: n_seg pairs (byte offset on the SSD, byte offset in the staging buffer); both are
// multiples of io_bytes. Segment s is split into sub_per_seg commands of io_bytes each.
__global__ void kado_bam_io(QueuePair* qps, uint32_t n_qps, const int64_t* desc, uint32_t n_seg,
                            uint32_t sub_per_seg, uint32_t io_bytes, uint32_t blk_log,
                            const uint64_t* prp1, const uint64_t* prp2, uint8_t opcode,
                            unsigned int* err_count, unsigned int* err_last) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_seg * sub_per_seg) return;
    const uint32_t s = t / sub_per_seg, k = t % sub_per_seg;
    const uint64_t dev = (uint64_t)desc[2 * s] + (uint64_t)k * io_bytes;
    const uint64_t stg = (uint64_t)desc[2 * s + 1] + (uint64_t)k * io_bytes;
    const uint64_t pi = stg / io_bytes;
    QueuePair* qp = qps + (t % n_qps);

    nvm_cmd_t cmd;
    const uint16_t cid = get_cid(&(qp->sq));
    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    nvm_cmd_data_ptr(&cmd, prp1[pi], prp2[pi]);
    nvm_cmd_rw_blks(&cmd, dev >> blk_log, (uint16_t)(io_bytes >> blk_log));
    (void)sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    const uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    // completion dword 3: [31:17] status (SCT/SC/M/DNR), [16] phase, [15:0] command id
    const uint32_t status = (((volatile nvm_cpl_t*)qp->cq.vaddr)[cq_pos].dword[3]) >> 17;
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    put_cid(&qp->sq, cid);
    if (status != 0) {
        atomicAdd(err_count, 1u);
        *(volatile unsigned int*)err_last = status;
    }
}

class Device {
  public:
    Device(const std::string& path, int cuda_device, int n_queues, int queue_depth)
        : cuda_device_(cuda_device) {
        check(cudaSetDevice(cuda_device), "cudaSetDevice");
        ctrl_ = new Controller(path.c_str(), 1, (uint32_t)cuda_device, (uint64_t)queue_depth, (uint64_t)n_queues);
        check(cudaHostAlloc((void**)&h_err_, 2 * sizeof(unsigned int), cudaHostAllocMapped), "cudaHostAlloc");
        h_err_[0] = h_err_[1] = 0;
        check(cudaHostGetDevicePointer((void**)&d_err_, h_err_, 0), "cudaHostGetDevicePointer");
    }

    ~Device() { close(); }

    py::dict info() const {
        require_open();
        py::dict d;
        d["block_size"] = (uint64_t)ctrl_->blk_size;
        d["num_blocks"] = (uint64_t)ctrl_->ns.size;
        d["capacity_bytes"] = (uint64_t)ctrl_->ns.size * ctrl_->blk_size;
        d["max_data_size"] = (uint64_t)ctrl_->info.max_data_size;
        d["page_size"] = (uint64_t)ctrl_->page_size;
        d["num_queue_pairs"] = (uint64_t)ctrl_->n_qps;
        d["io_bytes"] = (uint64_t)io_bytes_;
        return d;
    }

    // Map [addr, addr + bytes) of GPU memory (64 KiB aligned) for the SSD's DMA and build one
    // PRP pair per io_bytes piece. io_bytes: power of two, a multiple of the controller page,
    // at most the controller's MDTS and at most 512 controller pages (one PRP list page).
    void bind(uint64_t addr, uint64_t bytes, uint64_t io_bytes) {
        require_open();
        if (stg_map_ != nullptr) throw std::runtime_error("staging already bound");
        const uint64_t page = ctrl_->page_size;
        if (addr % (1u << 16) || bytes % (1u << 16)) throw std::runtime_error("staging must be 64 KiB aligned");
        if (io_bytes == 0 || io_bytes % page || bytes % io_bytes || (io_bytes & (io_bytes - 1)))
            throw std::runtime_error("io_bytes must be a power-of-two multiple of the controller page dividing the staging size");
        if (ctrl_->info.max_data_size && io_bytes > ctrl_->info.max_data_size)
            throw std::runtime_error("io_bytes exceeds the controller MDTS");
        if (io_bytes / page > 512) throw std::runtime_error("io_bytes needs more than one PRP list page");
        check(cudaSetDevice(cuda_device_), "cudaSetDevice");
        int st = nvm_dma_map_device(&stg_map_, ctrl_->ctrl, (void*)addr, bytes);
        if (!nvm_ok(st)) {
            stg_map_ = nullptr;
            throw std::runtime_error(std::string("nvm_dma_map_device: ") + nvm_strerror(st));
        }
        const uint64_t n_io = bytes / io_bytes, ppi = io_bytes / page;
        std::vector<uint64_t> p1(n_io), p2(n_io, 0);
        // PRP list for io i (when ppi > 2): ppi - 1 entries in a slot of list_slot bytes; slots
        // are a power of two <= one page, so no list crosses a controller page.
        uint64_t list_slot = 8;
        while (list_slot < (ppi - 1) * 8) list_slot <<= 1;
        std::vector<uint64_t> lists;
        if (ppi > 2) {
            const uint64_t list_bytes = ((n_io * list_slot + (1u << 16) - 1) >> 16) << 16;
            prp_mem_ = createDma(ctrl_->ctrl, list_bytes, cuda_device_);
            lists.assign(list_bytes / 8, 0);
        }
        for (uint64_t i = 0; i < n_io; i++) {
            const uint64_t first = i * ppi;
            p1[i] = stg_map_->ioaddrs[first];
            if (ppi == 2) {
                p2[i] = stg_map_->ioaddrs[first + 1];
            } else if (ppi > 2) {
                const uint64_t off = i * list_slot;
                for (uint64_t j = 1; j < ppi; j++) lists[off / 8 + (j - 1)] = stg_map_->ioaddrs[first + j];
                p2[i] = prp_mem_->ioaddrs[off / page] + off % page;
            }
        }
        if (ppi > 2)
            check(cudaMemcpy(prp_mem_->vaddr, lists.data(), lists.size() * 8, cudaMemcpyHostToDevice), "copy PRP lists");
        check(cudaMalloc((void**)&d_prp1_, n_io * 8), "cudaMalloc prp1");
        check(cudaMalloc((void**)&d_prp2_, n_io * 8), "cudaMalloc prp2");
        check(cudaMemcpy(d_prp1_, p1.data(), n_io * 8, cudaMemcpyHostToDevice), "copy prp1");
        check(cudaMemcpy(d_prp2_, p2.data(), n_io * 8, cudaMemcpyHostToDevice), "copy prp2");
        stg_bytes_ = bytes;
        io_bytes_ = (uint32_t)io_bytes;
    }

    // desc: int64 CUDA tensor [n_seg, 2] = (SSD byte offset, staging byte offset) per segment of
    // seg_bytes. Launches the I/O kernel on `stream` (a cudaStream_t as an integer).
    void submit(torch::Tensor desc, uint64_t seg_bytes, bool write, uint64_t stream) {
        require_open();
        if (stg_map_ == nullptr) throw std::runtime_error("staging not bound");
        TORCH_CHECK(desc.is_cuda() && desc.scalar_type() == torch::kInt64 && desc.is_contiguous() &&
                    desc.dim() == 2 && desc.size(1) == 2, "desc must be a contiguous int64 CUDA tensor [n, 2]");
        if (seg_bytes % io_bytes_) throw std::runtime_error("segment size is not a multiple of io_bytes");
        const uint32_t n_seg = (uint32_t)desc.size(0);
        if (n_seg == 0) return;
        const uint32_t sub = (uint32_t)(seg_bytes / io_bytes_);
        const uint32_t n = n_seg * sub, threads = 64, blocks = (n + threads - 1) / threads;
        kado_bam_io<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            ctrl_->d_qps, ctrl_->n_qps, desc.data_ptr<int64_t>(), n_seg, sub, io_bytes_, ctrl_->blk_size_log,
            d_prp1_, d_prp2_, write ? (uint8_t)NVM_IO_WRITE : (uint8_t)NVM_IO_READ, d_err_, d_err_ + 1);
        check(cudaGetLastError(), "kado_bam_io launch");
    }

    // (number of failed NVMe commands so far, NVMe status of the last failure)
    py::tuple errors() const {
        return py::make_tuple(((volatile unsigned int*)h_err_)[0], ((volatile unsigned int*)h_err_)[1]);
    }

    void close() {
        if (ctrl_ == nullptr) return;
        cudaSetDevice(cuda_device_);
        cudaDeviceSynchronize();
        if (d_prp1_) cudaFree(d_prp1_);
        if (d_prp2_) cudaFree(d_prp2_);
        d_prp1_ = d_prp2_ = nullptr;
        prp_mem_.reset();
        if (stg_map_) nvm_dma_unmap(stg_map_);
        stg_map_ = nullptr;
        delete ctrl_;
        ctrl_ = nullptr;
        if (h_err_) cudaFreeHost(h_err_);
        h_err_ = nullptr;
    }

  private:
    void require_open() const {
        if (ctrl_ == nullptr) throw std::runtime_error("device closed");
    }

    int cuda_device_;
    Controller* ctrl_ = nullptr;
    nvm_dma_t* stg_map_ = nullptr;
    DmaPtr prp_mem_;
    uint64_t* d_prp1_ = nullptr;
    uint64_t* d_prp2_ = nullptr;
    uint64_t stg_bytes_ = 0;
    uint32_t io_bytes_ = 0;
    unsigned int* h_err_ = nullptr;
    unsigned int* d_err_ = nullptr;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "kado BaM GPU-direct I/O engine";
    py::class_<Device>(m, "Device")
        .def(py::init<const std::string&, int, int, int>(), py::arg("path"), py::arg("cuda_device"),
             py::arg("num_queues") = 16, py::arg("queue_depth") = 1024)
        .def("info", &Device::info)
        .def("bind", &Device::bind, py::arg("addr"), py::arg("bytes"), py::arg("io_bytes"))
        .def("submit", &Device::submit, py::arg("desc"), py::arg("seg_bytes"), py::arg("write"), py::arg("stream"))
        .def("errors", &Device::errors)
        .def("close", &Device::close);
}
