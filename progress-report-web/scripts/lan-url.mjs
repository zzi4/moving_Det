import { isIPv4 } from "node:net";
import { networkInterfaces } from "node:os";
import { pathToFileURL } from "node:url";

const privateIpv4 =
  /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/;
const virtualInterface = /^(br-|docker|veth|virbr|tap|tun)/;
const unusableAddress =
  /^(127\.|169\.254\.|0\.|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)/;

export function chooseLanAddress(interfaces) {
  const physical = [];
  for (const [name, addresses] of Object.entries(interfaces)) {
    for (const item of addresses ?? []) {
      const isIpv4 = item.family === "IPv4" || item.family === 4;
      if (!isIpv4 || item.internal || unusableAddress.test(item.address)) {
        continue;
      }
      if (!virtualInterface.test(name)) physical.push(item.address);
    }
  }

  return (
    physical.find((address) => privateIpv4.test(address)) ?? null
  );
}

export function resolveLanAddress(interfaces, override) {
  if (!override) return chooseLanAddress(interfaces);
  if (!isIPv4(override)) {
    throw new Error(`MOVING_DET_LAN_HOST=${override} 不是有效的 IPv4 地址`);
  }
  const assigned = Object.values(interfaces)
    .flatMap((addresses) => addresses ?? [])
    .some((item) => item.address === override);
  if (!assigned) {
    throw new Error(
      `MOVING_DET_LAN_HOST=${override} 不属于本机任何网络接口`,
    );
  }
  return override;
}

export function isPrivateLanAddress(address) {
  return privateIpv4.test(address);
}

function printUrls() {
  const port = 8787;
  const address = resolveLanAddress(
    networkInterfaces(),
    process.env.MOVING_DET_LAN_HOST,
  );
  if (address) {
    console.log(`本机与局域网访问：http://${address}:${port}`);
    if (!isPrivateLanAddress(address)) {
      console.log(
        "警告：该地址不是 RFC1918 私有地址；请确认校园网/防火墙只允许可信设备访问。",
      );
    }
  } else {
    console.log(
      "未发现物理网卡上的 RFC1918 地址；如需使用校园网地址，请显式设置 MOVING_DET_LAN_HOST。",
    );
  }
}

const isMain =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMain) printUrls();
