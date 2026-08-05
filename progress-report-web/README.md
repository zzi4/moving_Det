# 航拍运动目标检测 POC 进展网页

这是现有阶段报告的局域网可视化版本。页面约每 10 秒刷新一次本机 calibration 状态，但不会启动、停止或修改实验。

## 启动

```bash
cd /home/stu1/Projects/moving_Det/.worktrees/motion-evidence-poc/progress-report-web
MOVING_DET_LAN_HOST=59.72.89.57 npm run lan
```

保持这个终端窗口运行。当前机器没有物理网卡上的 RFC1918 地址，因此必须显式指定受控校园网地址；服务只绑定该网卡的 `8787` 端口，不绑定 Docker 网桥。

另开一个终端查看访问地址：

```bash
MOVING_DET_LAN_HOST=59.72.89.57 npm run lan:url
```

同一校园网内的设备使用脚本输出的地址。`59.72.89.57` 不是 RFC1918 私有地址，启动脚本会显示暴露风险警告；应通过校园网边界或主机防火墙确保只有可信设备能够访问。

## 停止

回到运行 `npm run lan` 的终端，按 `Ctrl+C`。

## 验证

```bash
npm test
```

验证内容包括状态采集、局域网地址选择、网页构建和服务端渲染的报告关键数字。
