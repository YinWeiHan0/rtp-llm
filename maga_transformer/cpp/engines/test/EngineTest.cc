#include "c10/util/intrusive_ptr.h"
#include "src/fastertransformer/core/Types.h"
#include "torch/all.h"

#include "maga_transformer/cpp/engines/NormalEngine.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/models/W.h"
#include "gmock/gmock-actions.h"
#include "gmock/gmock-function-mocker.h"
#include "gtest/gtest.h"
#include <memory>

using namespace std;
namespace W  = ft::W;
namespace ft = fastertransformer;
namespace rtp_llm {

class NormalEngineTest: public DeviceTestBase {
public:

};

TEST_F(NormalEngineTest, testSimple) {
    rtp_llm::MagaInitParams rtp_llm_params;
    rtp_llm_params.gpt_init_parameter = c10::make_intrusive<GptInitParameter>(2, 64, 2, 20, 20, 128);
    // GptInitParameter params(2, 64, 2, 20, 20, 128);
    auto& params        = *rtp_llm_params.gpt_init_parameter;
    params.head_num_kv_ = 2;

    const size_t inter_size    = 512;
    params.inter_size_         = inter_size;
    params.inter_padding_size_ = inter_size;
    params.seq_size_per_block_ = 8;
    typedef half         T;
    const at::ScalarType scalar_type  = at::ScalarType::Half;
    const ft::DataType   data_type    = getTensorType<T>();
    auto                 mem_type     = ft::MemoryType::MEMORY_GPU;
    const size_t         hidden_units = 128;
    auto data = device_->allocateBuffer({data_type, {inter_size, inter_size}, AllocationType::DEVICE}, {});

    auto word_embeddings =
        make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{(size_t)20, hidden_units}, data->data());
    auto lm_head =
        make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{hidden_units, (size_t)20}, data->data());
    std::unordered_map<std::string, ft::ConstBufferPtr> global_weights;
    global_weights.emplace(W::embedding, std::move(word_embeddings));
    global_weights.emplace(W::lm_head, std::move(lm_head));

    std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layer_weights;
    for (int i = 0; i < params.num_layers_; ++i) {
        auto pre_layernorm_weights =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto pre_layernorm_beta =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto post_layernorm_weights =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto post_layernorm_beta =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto qkv_weights = make_unique<const ft::Buffer>(
            mem_type, data_type, vector<size_t>{hidden_units, 3, hidden_units}, data->data());
        auto qkv_weights_b = make_unique<const ft::Buffer>(
            mem_type, data_type, vector<size_t>{hidden_units, 3, hidden_units}, data->data());
        auto attention_layernorm =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto attention_layernorm_beta =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto attention_output_weight = make_unique<const ft::Buffer>(
            mem_type, data_type, vector<size_t>{hidden_units, hidden_units}, data->data());
        auto attention_output_weight_beta = make_unique<const ft::Buffer>(
            mem_type, data_type, vector<size_t>{hidden_units, hidden_units}, data->data());
        auto ffn_weight =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{hidden_units, inter_size}, data->data());
        auto ffn_weight_beta =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{hidden_units, inter_size}, data->data());
        auto ffn_output_weight =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{inter_size, hidden_units}, data->data());
        auto ffn_output_weight_beta =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{inter_size, hidden_units}, data->data());
        auto ffn_layer_norm =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{inter_size}, data->data());
        auto ffn_layer_norm_beta =
            make_unique<const ft::Buffer>(mem_type, data_type, vector<size_t>{inter_size}, data->data());
        std::unordered_map<std::string, ft::ConstBufferPtr> __weights;
        __weights.emplace(W::pre_ln_gamma, std::move(pre_layernorm_weights));
        __weights.emplace(W::pre_ln_beta, std::move(pre_layernorm_beta));
        __weights.emplace(W::attn_qkv_w, std::move(qkv_weights));
        __weights.emplace(W::attn_qkv_b, std::move(qkv_weights_b));
        __weights.emplace(W::attn_ln_gamma, std::move(attention_layernorm));
        __weights.emplace(W::attn_ln_beta, std::move(attention_layernorm_beta));
        __weights.emplace(W::attn_o_w, std::move(attention_output_weight));
        __weights.emplace(W::attn_o_b, std::move(attention_output_weight_beta));
        __weights.emplace(W::post_ln_gamma, std::move(post_layernorm_weights));
        __weights.emplace(W::post_ln_beta, std::move(post_layernorm_beta));
        __weights.emplace(W::ffn_w1, std::move(ffn_weight));
        __weights.emplace(W::ffn_b1, std::move(ffn_weight_beta));
        __weights.emplace(W::ffn_w2, std::move(ffn_output_weight));
        __weights.emplace(W::ffn_b2, std::move(ffn_output_weight_beta));
        __weights.emplace(W::ffn_ln_gamma, std::move(ffn_layer_norm));
        __weights.emplace(W::ffn_ln_beta, std::move(ffn_layer_norm_beta));
        layer_weights.push_back(std::move(__weights));
    }
    std::shared_ptr<GenerateInput> query   = make_shared<GenerateInput>();
    query->input_ids                       = createBuffer<int32_t>({7}, {1, 2, 3, 4, 5, 6, 7}, AllocationType::HOST);
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 3;
    shared_ptr<GenerateStream> stream      = make_shared<GenerateStream>(query);

    NormalEngine engine(rtp_llm_params, layer_weights, global_weights);
    ASSERT_TRUE(engine.startLoop().ok());
    ASSERT_TRUE(engine.enqueue(stream).ok());
    auto output1 = stream->nextOutput();
    ASSERT_TRUE(output1.ok());
    ASSERT_EQ(output1.value().aux_info.output_len, 1);

    auto output2 = stream->nextOutput();
    ASSERT_TRUE(output2.ok());
    ASSERT_EQ(output2.value().aux_info.output_len, 2);

    auto output3 = stream->nextOutput();
    ASSERT_TRUE(output3.ok());
    ASSERT_EQ(output3.value().aux_info.output_len, 3);

    ASSERT_TRUE(stream->finished());
    auto output4 = stream->nextOutput();
    ASSERT_TRUE(!output4.ok());
}

}  // namespace rtp_llm
