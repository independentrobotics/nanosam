# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import find_packages, setup

setup(
    name="nanosam",
    version="1.0",
    description="A distilled Segment Anything model variant capable of running in real-time on NVIDIA Jetson platforms with TensorRT.",
    author="Originally authored by NVIDIA, modifications made by Independent Robotics.",
    maintainer="Michael Fulton",
    maintainer_email="michael.fulton@independentrobotics.com",
    license="Apache 2.0",
    install_requires=[],
    packages=find_packages(),
)
    