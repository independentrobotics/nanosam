TARGET=$1
IR_UTILS=$HOME/code/ir_utils/
NANOSAM=$HOME/code/nanosam/

docker build --tag nanosam --file ./nanosam.Dockerfile \
    --target=$TARGET \
    --build-context ir_utils=$IR_UTILS \
    --build-context nanosam=$NANOSAM \
    .