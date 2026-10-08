module SOSDCUDAExt

using SOSD, CUDA

# Resident threads of the device: SMs × max threads per SM (e.g. T4 40×1024, A100 108×2048)
function SOSD.resident_threads(::CUDABackend)
    dev = CUDA.device()
    return CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) *
           CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
end

SOSD.available_memory(::CUDABackend) = Int(CUDA.available_memory())

end
