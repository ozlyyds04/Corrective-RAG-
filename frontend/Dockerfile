FROM node:20-alpine AS build

WORKDIR /app
# 连同 lock 文件一起拷贝并用 npm ci，保证构建可复现
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
