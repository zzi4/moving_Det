import { networkInterfaces } from "node:os";
import { pathToFileURL } from "node:url";

const privateIpv4 =
  /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/;
const virtualInterface = /^(br-|docker|veth|virbr|tap|tun)/;
const unusableAddress =
  /^(127\.|169\.254\.|0\.|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)/;

export function chooseLanAddress(interfaces) {
  const physical = [];
  const virtual = [];

  for (const [name, addresses] of Object.entries(interfaces)) {
    for (const item of addresses ?? []) {
      const isIpv4 = item.family === "IPv4" || item.family === 4;
      if (!isIpv4 || item.internal || unusableAddress.test(item.address)) {
        continue;
      }
      const target = virtualInterface.test(name) ? virtual : physical;
      target.push(item.address);
    }
  }

  return (
    physical.find((address) => privateIpv4.test(address)) ??
    physical[0] ??
    virtual.find((address) => privateIpv4.test(address)) ??
    null
  );
}

function printUrls() {
  const port = 8787;
  const address = chooseLanAddress(networkInterfaces());
  console.log(`本机访问：http://127.0.0.1:${port}`);
  if (address) {
    console.log(`局域网访问：http://${address}:${port}`);
  } else {
    console.log("未发现可用的局域网 IPv4 地址。");
  }
}

const isMain =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMain) printUrls();
