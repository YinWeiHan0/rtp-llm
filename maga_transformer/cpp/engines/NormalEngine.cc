#include "maga_transformer/cpp/executors/NormalExecutor.h"
#include "maga_transformer/cpp/engines/NormalEngine.h"
#include "maga_transformer/cpp/batch_stream_processor/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/schedulers/FIFOScheduler.h"
#include "src/fastertransformer/core/Types.h"

using namespace std;
namespace rtp_llm {

NormalEngine::NormalEngine(const MagaInitParams&                                                   params,
                           const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                           const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights) {
    executor_.reset(new NormalExecutor(params, layer_weights, weights));
    // TODO(xinfei.sxf) cache config from where, new cache Manager
    int   block_num     = 100;
    char* block_num_env = std::getenv("BLOCK_NUM");
    if (block_num_env) {
        block_num = std::stoi(block_num_env);
    }
    CacheConfig                   cache_config(params.gpt_init_parameter->num_layers_,
                                               block_num,
                                               params.gpt_init_parameter->head_num_kv_,
                                               params.gpt_init_parameter->size_per_head_,
                                               params.gpt_init_parameter->seq_size_per_block_,
                                               ft::DataType::TYPE_FP16);
    ncclComm_t                    nccl_op;
    ft::DeviceBase*               device        = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    std::shared_ptr<CacheManager> cache_manager = make_shared<CacheManager>(cache_config, nccl_op, device);
    scheduler_.reset(new FIFOScheduler(params, cache_manager));
}

NormalEngine::~NormalEngine() {
    (void)stop();
}

absl::Status NormalEngine::startLoop() {
    running_     = true;
    loop_thread_ = std::thread(&NormalEngine::loop, this);
    loop_thread_.detach();
    return absl::OkStatus();
}

absl::Status NormalEngine::stop() {
    running_ = false;
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    return absl::OkStatus();
}

void NormalEngine::loop() {
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            FT_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

absl::Status NormalEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

absl::Status NormalEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    FT_LOG_DEBUG("enqueue stream: %s", stream->debugString().c_str());
    return scheduler_->enqueue(stream);
}

absl::Status NormalEngine::step() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto streams_status = scheduler_->schedule();
    RETURN_IF_STATUS_OR_ERROR(streams_status);
    const auto& streams = streams_status.value();
    // FT_LOG_DEBUG("schedule res: %s", streams.debugString().c_str());
    if (streams.empty()) {
        // TODO(xinfei.sxf) 加一个notify的机制，防止busy polling，测试空转cpu。
        // std::this_thread::sleep_for(std::chrono::microseconds(1));
        FT_LOG_DEBUG("no query run and sleep");
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return absl::OkStatus();
    }
    return executor_->process(streams);
}

}  // namespace rtp_llm
