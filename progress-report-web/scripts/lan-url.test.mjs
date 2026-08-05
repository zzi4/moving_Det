import assert from "node:assert/strict";
import test from "node:test";

import {
  chooseLanAddress,
  resolveLanAddress,
} from "./lan-url.mjs";

test("prefers a private IPv4 address", () => {
  const address = chooseLanAddress({
    lo: [
      {
        address: "127.0.0.1",
        family: "IPv4",
        internal: true,
      },
    ],
    eno1: [
      {
        address: "192.168.1.24",
        family: "IPv4",
        internal: false,
      },
    ],
  });

  assert.equal(address, "192.168.1.24");
});

test("accepts the full RFC1918 IPv4 ranges", () => {
  assert.equal(
    chooseLanAddress({
      eth0: [
        { address: "172.31.9.4", family: "IPv4", internal: false },
      ],
    }),
    "172.31.9.4",
  );
  assert.equal(
    chooseLanAddress({
      eth0: [{ address: "10.8.0.6", family: 4, internal: false }],
    }),
    "10.8.0.6",
  );
});

test("returns null when no private IPv4 address is available", () => {
  assert.equal(
    chooseLanAddress({
      lo: [
        {
          address: "127.0.0.1",
          family: "IPv4",
          internal: true,
        },
      ],
      eth0: [
        {
          address: "203.0.113.2",
          family: "IPv4",
          internal: false,
        },
      ],
    }),
    null,
  );
});

test("does not advertise a public address or Docker bridge by default", () => {
  assert.equal(
    chooseLanAddress({
      eno1: [
        { address: "169.254.89.156", family: "IPv4", internal: false },
      ],
      eno2: [
        { address: "59.72.89.57", family: "IPv4", internal: false },
      ],
      "br-28c3f73dc74c": [
        { address: "172.19.0.1", family: "IPv4", internal: false },
      ],
      docker0: [
        { address: "172.17.0.1", family: "IPv4", internal: false },
      ],
    }),
    null,
  );
});

test("accepts a public campus address only through an explicit override", () => {
  const interfaces = {
    eno2: [
      { address: "59.72.89.57", family: "IPv4", internal: false },
    ],
  };

  assert.equal(
    resolveLanAddress(interfaces, "59.72.89.57"),
    "59.72.89.57",
  );
  assert.throws(
    () => resolveLanAddress(interfaces, "not-an-ip"),
    /不是有效的 IPv4/,
  );
});
