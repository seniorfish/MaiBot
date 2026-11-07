import { useEffect, useState } from 'react'

import type { StageStatusInfo } from './use-maisaka-monitor'

/** 后端的阶段标识与旧文案都在展示层统一成 Agent 术语。 */
export function formatAgentStageName(stage: string): string {
  const normalizedStage = stage.trim()
  const lowerStage = normalizedStage.toLowerCase()
  if (lowerStage === 'planner' || normalizedStage === '规划器') return 'Agent 决策'
  if (lowerStage === 'replyer' || normalizedStage === '回复器') return 'Agent 回复'
  if (normalizedStage === '等待消息') return '空闲'
  return normalizedStage || '未知阶段'
}

/** 后端 detail 文案可能仍带旧的 Planner/Replyer 字样，展示时一并替换。 */
export function formatAgentStatusText(text: string): string {
  return text
    .replace(/规划器/g, 'Agent')
    .replace(/回复器/g, 'Agent')
    .replace(/\bPlanner\b/gi, 'Agent')
    .replace(/\bReplyer\b/gi, 'Agent')
}

export function StageStatusLabel({ status }: { status: StageStatusInfo }) {
  const [now, setNow] = useState(() => Date.now() / 1000)
  const sleeping = status.sleeping === true
  const sleepUntil =
    sleeping && typeof status.sleepUntil === 'number' ? status.sleepUntil : undefined
  const waitUntil = status.agentState === 'wait' ? status.waitUntil : undefined
  const countdownUntil = sleepUntil ?? waitUntil

  useEffect(() => {
    if (countdownUntil === undefined) return
    setNow(Date.now() / 1000)
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => window.clearInterval(timer)
  }, [countdownUntil])

  // 新字段优先；字段缺失或为 null 时回落到原有的阶段/等待展示。
  if (sleeping) {
    return (
      <>
        待机中
        {sleepUntil !== undefined && ` (剩余 ${Math.max(0, Math.ceil(sleepUntil - now))}s)`}
      </>
    )
  }
  if (typeof status.eventQueueSize === 'number') {
    return <>待处理事件 {status.eventQueueSize}</>
  }

  // 使用后端的等待截止时间，刷新或重新进入页面后仍显示真实剩余时间。
  if (waitUntil !== undefined) {
    return <>等待({Math.max(0, Math.ceil(waitUntil - now))}s)</>
  }
  return <>{formatAgentStageName(status.stage)}</>
}
