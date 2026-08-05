import { spawn } from "node:child_process";
import { networkInterfaces } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  isPrivateLanAddress,
  resolveLanAddress,
} from "./lan-url.mjs";

const projectPath = dirname(dirname(fileURLToPath(import.meta.url)));
const address = resolveLanAddress(
  networkInterfaces(),
  process.env.MOVING_DET_LAN_HOST,
);

if (!address) {
  console.error(
    "未发现物理网卡上的 RFC1918 地址。若这是受控校园网，请显式运行：\n" +
      "MOVING_DET_LAN_HOST=<本机地址> npm run lan",
  );
  process.exit(2);
}

if (!isPrivateLanAddress(address)) {
  console.warn(
    `警告：正在绑定非 RFC1918 地址 ${address}。请确认防火墙只允许可信设备访问。`,
  );
}

const command = join(projectPath, "node_modules", ".bin", "vinext");
const child = spawn(
  command,
  ["dev", "--hostname", address, "--port", "8787"],
  {
    cwd: projectPath,
    env: {
      ...process.env,
      WRANGLER_LOG_PATH: ".wrangler/wrangler.log",
    },
    stdio: "inherit",
  },
);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}

child.on("exit", (code, signal) => {
  process.exitCode = signal ? 1 : (code ?? 1);
});
