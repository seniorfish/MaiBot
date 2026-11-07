import { describe, expect, it, vi } from 'vitest'

import { importJargons } from '@/lib/jargon-api'
import { importLearningCandidates, mergeLearningCandidates } from '@/lib/text-learning-api'

vi.mock('@/lib/jargon-api', () => ({ importJargons: vi.fn() }))

const candidate = { title: '起飞', description: '进展顺利', evidence: '这波起飞' }

describe('长文学习导入', () => {
  it('黑话导入复用真实聊天列表并跳过重复，不覆盖已有记录', async () => {
    await importLearningCandidates(['real-session', 'another-session'], [candidate])
    expect(importJargons).toHaveBeenCalledWith({
      target_session_ids: ['real-session', 'another-session'],
      conflict_strategy: 'skip',
      jargons: [
        expect.objectContaining({
          content: '起飞',
          meaning: '进展顺利',
          created_by: 'MANUAL',
          is_global: false,
        }),
      ],
    })
  })

  it('缺少用户选项时禁止入库', async () => {
    await expect(importLearningCandidates([], [candidate])).rejects.toThrow('请选择')
    await expect(importLearningCandidates(['session'], [])).rejects.toThrow('请选择')
    expect(importJargons).not.toHaveBeenCalled()
  })

  it('跨段去重按黑话词条合并同名词条', () => {
    const differentMeaning = { ...candidate, description: '非常高兴' }
    expect(mergeLearningCandidates([candidate], [candidate, differentMeaning])).toEqual([
      candidate,
    ])
  })

  it('黑话全局导入无需聊天，无原文也可以导入', async () => {
    await importLearningCandidates(
      [],
      [{ title: '全局黑话', description: '编辑含义' }],
      true
    )
    expect(importJargons).toHaveBeenCalledWith({
      target_session_ids: [],
      is_global: true,
      conflict_strategy: 'skip',
      jargons: [
        expect.objectContaining({ content: '全局黑话', meaning: '编辑含义', is_global: true }),
      ],
    })
  })

  it('全局导入拒绝混入聊天目标，所选内容为空时阻止入库', async () => {
    await expect(importLearningCandidates(['session'], [candidate], true)).rejects.toThrow(
      '不能同时指定聊天'
    )
    await expect(
      importLearningCandidates([], [{ title: ' ', description: '含义' }], true)
    ).rejects.toThrow('请填写')
    expect(importJargons).not.toHaveBeenCalled()
  })
})
