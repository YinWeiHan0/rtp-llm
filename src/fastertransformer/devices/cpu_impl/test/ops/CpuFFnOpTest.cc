#include "src/fastertransformer/devices/base_tests/FfnLayerTest.hpp"
#include "src/fastertransformer/devices/cpu_impl/CpuDevice.h"


class CpuFfnLayerTest: public FfnLayerTest {};

TEST_F(CpuFfnLayerTest, Gate_Fp16_FfnOpTest) {
    FfnOpTest(4, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP32);
    FfnOpTest(4, 2048, 4096, ActivationType::Swiglu, DataType::TYPE_FP32);
    FfnOpTest(128, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP32);
    FfnOpTest(1000, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP32);
    FfnOpTest(1, 2, 4096, ActivationType::Swiglu, DataType::TYPE_FP32);
    FfnOpTest(1000, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP32);
}

TEST_F(CpuFfnLayerTest, NoGate_Fp16_FfnOpTest) {
    FfnOpTest(4, 2048, 128, ActivationType::Geglu, DataType::TYPE_FP32);
    FfnOpTest(4, 2048, 4096, ActivationType::Geglu, DataType::TYPE_FP32);
    FfnOpTest(128, 2048, 128, ActivationType::Geglu, DataType::TYPE_FP32);
    FfnOpTest(1000, 2048, 128, ActivationType::Geglu, DataType::TYPE_FP32);
    FfnOpTest(1, 2, 4096, ActivationType::Geglu, DataType::TYPE_FP32);
    FfnOpTest(1000, 2048, 128, ActivationType::Geglu, DataType::TYPE_FP32);
}

