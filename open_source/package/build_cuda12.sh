set -x;

BASE_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_transformer_base_gpu_cuda12
TARGET_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_transformer_gpu_cuda12
#DEV_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_transformer_buildfarm_gpu_cuda12
DEV_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_transformer_dev_gpu_cuda12
BAZEL_ARGS="--config=cuda12"

sh package_docker.sh ${BASE_IMAGE} ${TARGET_IMAGE} ${DEV_IMAGE} ${BAZEL_ARGS}
