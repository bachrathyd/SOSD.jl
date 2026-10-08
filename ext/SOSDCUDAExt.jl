module SOSDCUDAExt

using SOSD, CUDA

# Resident threads of the device: SMs × max threads per SM (e.g. T4 40×1024, A100 108×2048)
function SOSD.resident_threads(::CUDABackend)
    dev = CUDA.device()
    return CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) *
           CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
end

# Free device memory. CUDA.jl's pool keeps freed buffers cached (invisible to the driver),
# so collect and release them first — otherwise the automatic batch size collapses to a
# few points after the first large batch.
function SOSD.available_memory(::CUDABackend)
    GC.gc(false)
    CUDA.reclaim()
    return Int(CUDA.available_memory())
end

end
