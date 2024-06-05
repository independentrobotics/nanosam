docker run \
    -it \
    --gpus all \
    --net=host \
    --ipc=host \
    --pid=host \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    --volume="$HOME/.config/independent-robotics.yaml:/root/.config/independent-robotics.yaml" \
    --volume="$HOME/IndependentRobotics:/root/IndependentRobotics" \
    --volume="$HOME/docker_out:/root/out" \
    nanosam:latest \
    /bin/bash