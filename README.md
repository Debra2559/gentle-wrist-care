# Gentle Care Wrist

我们想为一个面向于有腱鞘炎（曾经有或者正在有）的女生提供一款智能护腕，想让你帮我做一下软件部分的开发（支持传数据到app），app支持展示相关的数据监测、预警、建议等等，你帮我做下这个网站，记得做移动端H5适配；风格我希望是治愈、简洁、温柔的如图

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://gentle-wrist-care.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/eb0b59ab-496e-4411-8bae-ad853b9d5211).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```

## algorithm/ —— SheWrist 腕部暴露算法后端

`algorithm/` 目录来自 [lxy2137/algorithm_for_predict](https://github.com/lxy2137/algorithm_for_predict)，是 Python 实现的离线分析与后端 API（腕角估计、暴露剂量、阈值状态机、影子 ML 与解释服务）。

本地启动：

```bash
cd algorithm
python3 -m pip install -r requirements.txt -r requirements-api.txt
PYTHONPATH=src python3 scripts/run_api.py --host 127.0.0.1 --port 8000
```

前端通过服务端环境变量对接（见 `src/lib/shewrist.functions.ts`）：

- `SHEWRIST_API_BASE_URL`：例如 `http://127.0.0.1:8000`
- `SHEWRIST_API_TOKEN`：可选 Bearer 令牌

未配置时「报告」页自动回退到结构一致的演示数据。接口规范见 `algorithm/docs/backend_api.md`。
