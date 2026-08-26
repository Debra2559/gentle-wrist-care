export type Severity = "calm" | "notice" | "warn";

export const todayStats = {
  wearMinutes: 312,
  wearGoal: 480,
  repetitiveActions: 4180,
  actionGoal: 5000,
  restBreaks: 5,
  restGoal: 8,
  painScore: 2.1,
  strainIndex: 38,
  temperature: 33.4,
  batteryPercent: 76,
  syncedAt: "10 分钟前",
};

export const weekTrend = [
  { day: "周一", strain: 52, actions: 5300, rest: 3 },
  { day: "周二", strain: 44, actions: 4700, rest: 5 },
  { day: "周三", strain: 61, actions: 6100, rest: 2 },
  { day: "周四", strain: 39, actions: 4200, rest: 6 },
  { day: "周五", strain: 47, actions: 5000, rest: 4 },
  { day: "周六", strain: 28, actions: 2600, rest: 7 },
  { day: "周日", strain: 38, actions: 4180, rest: 5 },
];

export const hourlyStrain = [
  { hour: "8", value: 12 },
  { hour: "10", value: 34 },
  { hour: "11", value: 58 },
  { hour: "13", value: 26 },
  { hour: "15", value: 71 },
  { hour: "16", value: 49 },
  { hour: "18", value: 33 },
  { hour: "20", value: 18 },
];

export const alerts: {
  id: string;
  title: string;
  detail: string;
  time: string;
  severity: Severity;
}[] = [
  {
    id: "a1",
    title: "连续打字 52 分钟",
    detail: "手腕屈伸频率偏高，建议现在停下来做一次 90 秒的舒展。",
    time: "今天 15:24",
    severity: "warn",
  },
  {
    id: "a2",
    title: "护腕佩戴略松",
    detail: "压力传感器读数下降，轻轻收紧一格会更稳定地支撑拇指侧。",
    time: "今天 11:10",
    severity: "notice",
  },
  {
    id: "a3",
    title: "上午状态很温和",
    detail: "劳损指数保持在安全区间，节奏刚刚好，继续保持。",
    time: "今天 09:40",
    severity: "calm",
  },
  {
    id: "a4",
    title: "夜间温度偏低",
    detail: "手腕表面温度 30.2°C，睡前可以开启 10 分钟低温热敷。",
    time: "昨天 23:05",
    severity: "notice",
  },
];

export const suggestions = [
  {
    id: "s1",
    tag: "此刻",
    title: "90 秒腕部舒展",
    body: "手掌向下轻压桌面，指尖朝向自己，缓慢呼吸 5 次；再翻转手掌重复一次。",
    minutes: 2,
  },
  {
    id: "s2",
    tag: "工作节奏",
    title: "45 分钟提醒一次",
    body: "把连续操作时间控制在 45 分钟内，起身走动或握放软球 10 次。",
    minutes: 1,
  },
  {
    id: "s3",
    tag: "夜间",
    title: "睡前热敷 + 固定",
    body: "40°C 温热毛巾覆盖 10 分钟，睡眠时保持护腕轻度固定，避免夜间弯折。",
    minutes: 10,
  },
  {
    id: "s4",
    tag: "长期",
    title: "拇指侧肌力练习",
    body: "用弹力圈做拇指外展，每组 12 次、每天 2 组，疼痛时暂停。",
    minutes: 6,
  },
];

export const recoveryNotes = [
  { date: "本周", text: "疼痛评分从 3.4 降到 2.1，休息次数增加了 2 次。" },
  { date: "本月", text: "高负荷时段减少 31%，夜间僵硬感记录 4 次（上月 9 次）。" },
];

export const severityStyles: Record<Severity, { label: string; dot: string; chip: string }> = {
  calm: { label: "安心", dot: "bg-calm", chip: "bg-calm/15 text-foreground" },
  notice: { label: "留意", dot: "bg-sky", chip: "bg-sky/20 text-foreground" },
  warn: { label: "预警", dot: "bg-warn", chip: "bg-warn/20 text-foreground" },
};

export const vibrationStages: {
  label: string;
  desc: string;
  pattern: number[];
  wave: number[];
}[] = [
  {
    label: "微语",
    desc: "第一层：像羽毛落下，仅在手腕内侧轻点两下，不打断你手上的事。",
    pattern: [60, 120, 60],
    wave: [18, 26, 20, 30, 22, 16, 12],
  },
  {
    label: "轻抚",
    desc: "第二层：三次渐强的短震，提示你可以放下鼠标做一次舒展。",
    pattern: [90, 120, 140, 120, 190],
    wave: [20, 34, 46, 58, 44, 30, 22],
  },
  {
    label: "坚定",
    desc: "第三层：连续波浪式加压震动，直到姿势回到安全区间才停。",
    pattern: [140, 100, 200, 100, 260, 100, 320],
    wave: [26, 44, 62, 80, 96, 70, 48],
  },
];

export const correctionLog = [
  { time: "15:26", text: "腕部背屈 34°，气囊缓慢充压 6 秒，已回到中立位。" },
  { time: "13:48", text: "拇指侧支撑略松，自动补压 2 格。" },
  { time: "10:12", text: "长时间下垂，托举支撑维持了 4 分钟。" },
];
