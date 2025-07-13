# Base image.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

# Build-time parameters.
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG REQUIREMENT_FILE
ARG DOTFILES_REPO

# Set up env and home directory.
ENV HOME=/home/$USERNAME
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="$HOME/.local/bin:$PATH"
ENV PYTHONPATH="$HOME/.local/lib/python3.10/site-packages"
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0;12.0+PTX"

# Basic packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo ca-certificates git curl python3 python3-dev \
    python3-pip python3-venv openssh-client less tzdata \
    zsh locales ninja-build
RUN sudo dpkg-reconfigure -f noninteractive tzdata
RUN locale-gen en_US.UTF-8


# Create group and user.
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m -s /usr/bin/zsh $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to the user.
USER $USERNAME
WORKDIR $HOME

# Install uv.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv venv --python 3.10 .venv

ENV UV_PROJECT_ENVIRONMENT=${HOME}/.venv

# Install torch.
RUN uv pip install --prefix ${HOME}/.local \
    torch==2.9.0.dev20250712+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128

COPY ${REQUIREMENT_FILE} ./requirements.lock
RUN uv pip install --prefix ${HOME}/.local \
    -r ./requirements.lock \
    --index-strategy unsafe-best-match --no-build-isolation -v

# Set up SSH.
RUN --mount=type=ssh,uid=${USER_UID},gid=${USER_GID} mkdir -p -m 700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Clone dotfiles repository.
RUN --mount=type=ssh,uid=${USER_UID},gid=${USER_GID} git clone --recurse-submodules "${DOTFILES_REPO}" "${HOME}/dotfiles"

# Install dotbot and run the configuration installation.
RUN uv tool install dotbot

# Split it into multiple steps.
RUN uv run dotbot -d "${HOME}/dotfiles" -c "${HOME}/dotfiles/install.conf.yaml" --only shell
RUN uv run dotbot -d "${HOME}/dotfiles" -c "${HOME}/dotfiles/install.conf.yaml" --only link

# Clean up apt cache.
RUN sudo rm -rf /var/lib/apt/lists/*

ENV MAX_JOBS=18
