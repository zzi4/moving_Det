# 航拍运动目标检测 POC 进展网页

这是现有阶段报告的局域网可视化版本。页面约每 10 秒刷新一次本机 calibration 状态，但不会启动、停止或修改实验。

## 启动

```bash
cd /home/stu1/Projects/moving_Det/.worktrees/motion-evidence-poc/progress-report-web
npm run lan
```

保持这个终端窗口运行。服务默认监听 `0.0.0.0:8787`。

另开一个终端查看访问地址：

```bash
npm run lan:url
```

本机使用 `http://127.0.0.1:8787`，同一局域网内的其他设备使用脚本输出的局域网地址。

## 停止

回到运行 `npm run lan` 的终端，按 `Ctrl+C`。

## 验证

```bash
npm test
```

验证内容包括状态采集、局域网地址选择、网页构建和服务端渲染的报告关键数字。
