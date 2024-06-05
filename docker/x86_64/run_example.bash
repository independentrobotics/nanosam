docker run \
    -it \
    --gpus all \
    --net=host \
    --ipc=host \
    --pid=host \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    --volume="$HOME/.config/independent-robotics.yaml:/root/.config/independent-robotics.yaml" \
    --volume="$HOME/IndependentRobotics:/root/IndependentRobotics" \
    --volume="$1:/root/out" \
    nanosam:latest \
    /usr/bin/python3 /root/example.py "${@:2}"