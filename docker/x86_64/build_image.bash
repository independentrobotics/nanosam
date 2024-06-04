TARGET=$1
IR_UTILS=/home/michael/code/ir_utils/
NANOSAM=/home/michael/code/nanosam/

docker build --tag nanosam --file ./nanosam.Dockerfile \
    --target=$TARGET \
    --build-context ir_utils=$IR_UTILS \
    --build-context nanosam=$NANOSAM \
    .