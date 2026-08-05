import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "航拍运动目标检测 POC｜阶段进展",
  description:
    "基于相邻帧运动证据的航拍小目标 OBB 检测、tubelet 与轨迹系统阶段报告。",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
