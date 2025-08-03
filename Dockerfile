# docker run  -v /mnt/e:/mnt/e   -it -e DATA_DRIVE=/mnt/e --gpus all -rm  rl:latest
# docker run  -v /mnt/e:/mnt/e   -it -e DATA_DRIVE=/mnt/e --gpus all  rl:latest
# docker save -o perf-test.tar rl:latest
# ------------------------
# 1) Build (SDK) stage

# ------------------------
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build

# bring in your repo
WORKDIR /src
COPY . .

# restore & publish only your app (self-contained will bundle the .NET runtime)
RUN --mount=type=secret,id=nugetconfig,target=/root/.nuget/NuGet/NuGet.Config \
    dotnet restore src/Model1/Model1.fsproj

WORKDIR /src/src/Model1
RUN --mount=type=secret,id=nugetconfig,target=/root/.nuget/NuGet/NuGet.Config \
    dotnet publish \
    --configuration Release \
    --runtime linux-x64 \
    --self-contained true \
    --output /app

# ------------------------
# 2) Final (runtime + CUDA) stage
# ------------------------
FROM ghcr.io/ggml-org/llama.cpp:full-cuda

WORKDIR /usr/local/share/ca-certificates
#COPY ./certs .I 
#COPY ./docker_extra/localhost-tm1.pfx .
RUN update-ca-certificates
RUN apt-get update && apt-get install -y wget apt-transport-https libicu70 libunwind8

# copy only the published/final bits from build stage
WORKDIR /app
COPY --from=build /app .

# launch the self-contained binary directly
ENTRYPOINT ["./Model1"]
