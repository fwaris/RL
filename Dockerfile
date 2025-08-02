# docker run  -v /mnt/e:/mnt/e   -it -e DATA_DRIVE=/mnt/e --gpus all -rm  rl:latest
# docker run  -v /mnt/e:/mnt/e   -it -e DATA_DRIVE=/mnt/e --gpus all  rl:latest
# docker save -o perf-test.tar rl:latest
FROM ghcr.io/ggml-org/llama.cpp:full-cuda AS gpu-base
#FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04 AS gpu-base
EXPOSE 57101
#ENV ASPNETCORE_URLS=http://*:57101

WORKDIR /usr/local/share/ca-certificates
#COPY ./certs .I 
#COPY ./docker_extra/localhost-tm1.pfx .
RUN update-ca-certificates

# Install prerequisites + ICU
RUN apt-get update && apt-get install -y wget apt-transport-https libicu70 libunwind8

# Install .NET SDK (latest LTS, e.g. 8.0)
RUN wget https://dot.net/v1/dotnet-install.sh && \
    bash dotnet-install.sh -Channel STS && \
    rm dotnet-install.sh


ENV DOTNET_ROOT=/root/.dotnet
ENV PATH=$DOTNET_ROOT:$PATH
RUN echo "Dotnet SDK version:" && dotnet --version

FROM gpu-base AS build

WORKDIR /src
COPY . .

RUN --mount=type=secret,id=nugetconfig,target=/root/.nuget/NuGet/NuGet.Config \
    dotnet restore

RUN --mount=type=secret,id=nugetconfig,target=/root/.nuget/NuGet/NuGet.Config \
    dotnet build --configuration Release
    
WORKDIR /src/src/Model1
RUN --mount=type=secret,id=nugetconfig,target=/root/.nuget/NuGet/NuGet.Config \
    dotnet publish  -o /perfTest -r linux-x64 --self-contained -c Release 

# RUN dotnet restore
# RUN dotnet publish  -o /perfTest -r linux-x64 --self-contained --project Model1

FROM build AS final

WORKDIR /perTest
COPY --from=build /perfTest .
ENTRYPOINT ["dotnet", "Model1.dll"]


