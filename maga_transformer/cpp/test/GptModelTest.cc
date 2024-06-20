#include "src/fastertransformer/devices/testing/TestBase.h"

#define private public

#include "maga_transformer/cpp/models/ModelFactory.h"
#include "maga_transformer/cpp/test/ModelTestUtil.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/devices/torch_impl/GptModel.hpp"

using namespace std;
using namespace rtp_llm;
using namespace fastertransformer;

class GptModelTest: public DeviceTestBase {
};

TEST_F(GptModelTest, testSimple) {
    const auto path = test_data_path_ + "../../test/model_test/fake_test/testdata/qwen_0.5b";
    auto weights = loadWeightsFromDir(path);
    assert(weights->lm_head->kernel);
    assert(weights->embedding);
    assert(weights->layers.size() == 24);

    GptModelDescription description;
    description.activation_type = ActivationType::Swiglu;
    description.norm_type = NormType::rmsnorm;
    auto& attention_conf = description.attention_conf;
    attention_conf.head_num = 16;
    attention_conf.kv_head_num = 16;
    attention_conf.size_per_head = 64;
    attention_conf.tokens_per_block = 16;
    attention_conf.rope_config.embedding_style = RopeType::Base;
    attention_conf.rope_config.embedding_dim = 64;
    attention_conf.rope_config.embedding_base = 1000000;
    attention_conf.mask_type = AttentionMaskType::causalMask;
    auto model = createGptModel({device_, *weights, description});

    const auto cache_block_num = 128;
    CacheConfig cache_config(
        weights->layers.size(),
        cache_block_num,
        attention_conf.kv_head_num,
        attention_conf.size_per_head,
        attention_conf.tokens_per_block,
        DataType::TYPE_FP16
    );

    const std::vector<int32_t> input_lengths_vec = {3};

    auto combo_tokens = createBuffer<int32_t>({3}, {13048, 11, 220}, AllocationType::HOST);
    auto input_lengths = createBuffer<int32_t>({1}, input_lengths_vec, AllocationType::HOST);
    auto sequence_lengths = createBuffer<int32_t>({0}, {}, AllocationType::HOST);
    auto kv_cache = torch::empty(0);
    auto kv_cache_blocks = allocateKVBlocks(cache_config, input_lengths_vec, kv_cache);
    const auto mask_tensor = create_context_mask(input_lengths_vec).to(torch::kFloat16);
    const auto mask_buf = tensorToBuffer(mask_tensor);

    GptModelInputs inputs = {
        std::move(combo_tokens), std::move(input_lengths), std::move(sequence_lengths)
    };
    inputs.lm_output_indexes = createBuffer<int32_t>({1}, {2}, AllocationType::HOST);
    inputs.attention_mask = mask_buf;
    inputs.kv_cache_blocks = kv_cache_blocks;
    device_->syncAndCheck();

    // temporarily disable test for cpu/arm device
    // enable this back when device is ready.
    if (DeviceFactory::getDefaultDevice()->getDeviceProperties().type != DeviceType::Cuda) {
        try {
            auto outputs = model->forward(inputs);
            printBufferData(*outputs.logits, "logits");
            auto output_tensor = bufferToTensor(*outputs.logits);
        } catch (const std::exception& e) {
            std::cout << e.what() << std::endl;
        }
        return;
    }

    auto outputs = model->forward(inputs);
    device_->syncAndCheck();
    printBufferData(*outputs.logits, "logits");
    auto output_tensor = bufferToTensor(*outputs.logits);

    // expected to output token 151645
    assertTensorClose(
        output_tensor.index({torch::indexing::Slice(), 151645}),
        bufferToTensor(*createBuffer<float>({1}, {15.7891}, AllocationType::HOST)),
        1e-1, 5e-2
    );
    assertTensorClose(
        output_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}),
        bufferToTensor(*createBuffer<float>({1, 3},
            {7.1562, -9.3672, -0.8486}, AllocationType::HOST)),
        0.2, 0.1
    );

    inputs.combo_tokens = createBuffer<int32_t>({1}, {151645}, AllocationType::HOST);
    inputs.input_lengths = createBuffer<int32_t>({1}, {3}, AllocationType::HOST);
    inputs.sequence_lengths = createBuffer<int32_t>({1}, {3}, AllocationType::HOST);
    inputs.lm_output_indexes = createBuffer<int32_t>({1}, {0}, AllocationType::HOST);
    device_->syncAndCheck();
    outputs = model->forward(inputs);
    device_->syncAndCheck();
    output_tensor = bufferToTensor(*outputs.logits);

    // expected output token 198
    assertTensorClose(
        output_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(190, 200)}),
        bufferToTensor(*createBuffer<float>(
            {10}, {1.1670, -0.6973, -0.6919, -1.7705, -1.4453,
                -0.9766, -0.1351,  2.6152, 28.2500, -0.1479}, AllocationType::HOST)),
        1e-1, 1e-2
    );

    // combining two previous queries.
    inputs.combo_tokens = createBuffer<int32_t>({4}, {151645, 13048, 11, 220}, AllocationType::HOST);
    inputs.input_lengths = createBuffer<int32_t>({2}, {3, 3}, AllocationType::HOST);
    inputs.sequence_lengths = createBuffer<int32_t>({1}, {3}, AllocationType::HOST);
    inputs.lm_output_indexes = createBuffer<int32_t>({2}, {0, 3}, AllocationType::HOST);

    inputs.kv_cache_blocks = allocateKVBlocks(cache_config, {3, 3}, kv_cache);
    for (auto layer_id = 0; layer_id < inputs.kv_cache_blocks->shape()[0]; layer_id++) {
        auto layer_src = kv_cache_blocks->view(layer_id, 1);
        auto layer_dst = inputs.kv_cache_blocks->view(layer_id, 1);
        auto reshaped_dst = layer_dst.reshape({2, 2, 2}); // [bs, 2, blocks]
        device_->copy({reshaped_dst.view(0, 1), layer_src.reshape({1, 2, 2})});
    }

    device_->syncAndCheck();
    outputs = model->forward(inputs);
    device_->syncAndCheck();
    output_tensor = bufferToTensor(*outputs.logits);
    assertTensorClose(
        output_tensor.index({torch::indexing::Slice(1, 2), 151645}),
        bufferToTensor(*createBuffer<float>({1}, {15.7891}, AllocationType::HOST)),
        1e-1, 5e-2
    );
    assertTensorClose(
        output_tensor.index({torch::indexing::Slice(1, 2), torch::indexing::Slice(0, 3)}),
        bufferToTensor(*createBuffer<float>({1, 3},
            {7.1562, -9.3672, -0.8486}, AllocationType::HOST)),
        0.2, 0.1
    );
    assertTensorClose(
        output_tensor.index({torch::indexing::Slice(0, 1), torch::indexing::Slice(190, 200)}),
        bufferToTensor(*createBuffer<float>(
            {10}, {1.1670, -0.6973, -0.6919, -1.7705, -1.4453,
                -0.9766, -0.1351,  2.6152, 28.2500, -0.1479}, AllocationType::HOST)),
        1e-1, 1e-2
    );

}

TEST_F(GptModelTest, testAttentionInputs) {
    GptModelDescription description;
    Weights weights;
    auto model = createGptModel({device_, weights, description});
    GptModelInputs inputs;
    inputs.kv_cache_blocks = createBuffer<int64_t>({1, 2, 1, 10}, std::vector<int64_t>(20, 0), AllocationType::HOST);
    inputs.input_lengths = createBuffer<int32_t>({4}, {3, 5, 2, 7}, AllocationType::HOST);
    inputs.sequence_lengths = createBuffer<int32_t>({0}, {}, AllocationType::HOST);
    inputs.combo_tokens = createBuffer<int32_t>({17}, std::vector<int32_t>(17, 0), AllocationType::HOST);
    AttentionCommonInputs attention_inputs({
            *inputs.input_lengths,
            *inputs.sequence_lengths
        });

    {
        device_->syncAndCheck();
        model->prepareAttentionInputs(inputs, attention_inputs);
        device_->syncAndCheck();
        printBuffer<int32_t>(*attention_inputs.cu_seqlens);
        printBuffer<int32_t>(*attention_inputs.padding_offset);
        assertBufferValueEqual<int32_t>(*attention_inputs.cu_seqlens, {0, 3, 8, 10, 17});
        assertBufferValueEqual<int32_t>(*attention_inputs.padding_offset,
            {0, 0, 0, 4, 4, 4, 4, 4, 6, 6, 11, 11, 11, 11, 11, 11, 11});
        ASSERT_EQ(attention_inputs.context_batch_size, 4);
        ASSERT_EQ(attention_inputs.context_max_seq_len, 7);
        ASSERT_EQ(attention_inputs.decoder_batch_size, 0);
        ASSERT_EQ(attention_inputs.decoder_max_seq_len, 0);
    }

    inputs.sequence_lengths = createBuffer<int32_t>({3}, {4, 19, 23}, AllocationType::HOST);
    inputs.combo_tokens = createBuffer<int32_t>({10}, std::vector<int32_t>(10, 0), AllocationType::HOST);
    {
        device_->syncAndCheck();
        model->prepareAttentionInputs(inputs, attention_inputs);
        device_->syncAndCheck();
        printBuffer<int32_t>(*attention_inputs.cu_seqlens);
        printBuffer<int32_t>(*attention_inputs.padding_offset);
        assertBufferValueEqual<int32_t>(*attention_inputs.cu_seqlens, {0, 7});
        assertBufferValueEqual<int32_t>(*attention_inputs.padding_offset,
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        ASSERT_EQ(attention_inputs.context_batch_size, 1);
        ASSERT_EQ(attention_inputs.context_max_seq_len, 7);
        ASSERT_EQ(attention_inputs.decoder_batch_size, 3);
        ASSERT_EQ(attention_inputs.decoder_max_seq_len, 23);
    }

    inputs.sequence_lengths = createBuffer<int32_t>({2}, {4, 6}, AllocationType::HOST);
    inputs.combo_tokens = createBuffer<int32_t>({11}, std::vector<int32_t>(11, 0), AllocationType::HOST);
    {
        device_->syncAndCheck();
        model->prepareAttentionInputs(inputs, attention_inputs);
        device_->syncAndCheck();
        printBuffer<int32_t>(*attention_inputs.cu_seqlens);
        printBuffer<int32_t>(*attention_inputs.padding_offset);
        assertBufferValueEqual<int32_t>(*attention_inputs.cu_seqlens, {0, 2, 9});
        assertBufferValueEqual<int32_t>(*attention_inputs.padding_offset,
            {0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0});
        ASSERT_EQ(attention_inputs.context_batch_size, 2);
        ASSERT_EQ(attention_inputs.context_max_seq_len, 7);
        ASSERT_EQ(attention_inputs.decoder_batch_size, 2);
        ASSERT_EQ(attention_inputs.decoder_max_seq_len, 6);
    }
}
