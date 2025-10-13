FROM denoland/deno:alpine

WORKDIR /app

# 复制所有文件到容器
COPY . .

# 直接缓存 main.ts，让 Deno 自动解析和处理所有依赖
RUN deno cache main.ts

EXPOSE 8001

ENV FACTORY_API_KEYS=""
ENV PROXY_ACCESS_KEYS=""

CMD ["run", "-A", "main.ts"]
