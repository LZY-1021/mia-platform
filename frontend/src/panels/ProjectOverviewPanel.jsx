import React from 'react'
import { Button, Col, Row, Tag, Typography } from 'antd'
import {
  ApiOutlined,
  ArrowRightOutlined,
  BarChartOutlined,
  DatabaseOutlined,
  FileSearchOutlined,
  PictureOutlined,
  RadarChartOutlined,
  SafetyCertificateOutlined,
  ScanOutlined,
  SyncOutlined,
  TeamOutlined,
} from '@ant-design/icons'

const { Paragraph, Title } = Typography

const copy = {
  zh: {
    eyebrow: 'MEMBERSHIP PRIVACY AUDIT',
    title: '让模型记忆风险变得可测、可见、可验证',
    lead: '成员推断攻击评测平台面向大语言模型与视觉语言模型，帮助数据方核验训练边界，也帮助模型方在发布前发现不应保留的隐私记忆。',
    enter: '进入检测控制台',
    flow: '平台如何形成证据',
    flowLead: '从数据与模型配置，到攻击评测和可解释结论，形成可复现实验闭环。',
    audience: '谁可以使用平台',
    dataOwner: '数据方：核验数据是否被不当记忆',
    dataOwnerDesc: '对候选模型执行标准化成员推断评测，比较成员与非成员样本的响应差异，为数据治理和风险沟通提供量化依据。',
    modelOwner: '模型方：在上线前完成隐私自检',
    modelOwnerDesc: '在不同模型、数据集和攻击设置下复现实验，定位记忆风险，并以统一指标检验缓解措施是否有效。',
    innovations: '平台创新点',
    innovationLead: '不只给出一个分数，而是把攻击方法、统计证据和可解释样本放在同一个评测流程中。',
    impact: '面向可信模型的隐私评测基础设施',
    impactLead: '平台支持从研究验证走向发布前审计、数据授权核验与持续风险监测，让成员推断结论更容易理解、复现与比较。',
    ready: '准备开始一次隐私评测？',
    readyLead: '选择模型与数据集，配置攻击参数，即可进入统一检测控制台。',
    steps: [
      ['配置对象', '导入或选择目标模型、成员样本与非成员样本。'],
      ['提取信号', '计算黑盒概率特征，白盒模式还读取内部表征。'],
      ['执行攻击', '按模型类型与实验设置调用相应的黑盒或白盒攻击方法。'],
      ['解释结论', '输出 AUC、显著性、校准结果和典型证据样本。'],
    ],
    innovationItems: [
      ['统一评估框架', '以模块化接口集成多种攻击方法、指标计算与结果可视化，实现标准化评测。'],
      ['预计算与在线评测解耦', '将耗时的特征准备与交互式评测分离，降低运行成本并提升实验响应效率。'],
      ['多模态模型覆盖', '同时面向 LLM 与 VLM 组织黑盒、白盒检测，支持不同模态任务的风险分析。'],
      ['多方法对比与可解释分析', '通过 ROC、指标表和证据样本横向比较攻击效果，让检测结论更透明可信。'],
    ],
  },
  en: {
    eyebrow: 'MEMBERSHIP PRIVACY AUDIT',
    title: 'Make model memorization risk measurable, visible, and verifiable',
    lead: 'This platform evaluates membership-inference risks for language and vision-language models. It helps data owners audit training boundaries and helps model owners find unintended memorization before release.',
    enter: 'Open Detection Console',
    flow: 'How the Platform Builds Evidence',
    flowLead: 'A reproducible path from model and data configuration to attack evaluation and explainable conclusions.',
    audience: 'Who the Platform Serves',
    dataOwner: 'Data owners: verify inappropriate memorization',
    dataOwnerDesc: 'Run standardized membership-inference evaluations on a candidate model and quantify response differences between member and non-member samples.',
    modelOwner: 'Model owners: test privacy before deployment',
    modelOwnerDesc: 'Reproduce experiments across models, datasets, and attack settings to locate memorization risks and validate mitigation measures.',
    innovations: 'What Makes It Different',
    innovationLead: 'The platform combines attack methods, statistical evidence, and interpretable examples in one evaluation workflow instead of reporting a single score.',
    impact: 'Privacy Evaluation Infrastructure for Trustworthy Models',
    impactLead: 'It supports research validation, pre-release audits, data-rights verification, and continuous risk monitoring with conclusions that are easier to compare and explain.',
    ready: 'Ready to Start a Privacy Evaluation?',
    readyLead: 'Select a model and dataset, configure the attack, and enter the unified detection console.',
    steps: [
      ['Configure targets', 'Select the target model plus member and non-member samples.'],
      ['Extract signals', 'Compute black-box probability features and white-box internal signals.'],
      ['Run attacks', 'Run the appropriate black-box or white-box attack for the selected model and experiment setup.'],
      ['Explain results', 'Review AUC, significance, calibration, and representative evidence.'],
    ],
    innovationItems: [
      ['Unified evaluation framework', 'Modular interfaces combine attack methods, metric computation, and result visualization into a standardized workflow.'],
      ['Decoupled precomputation and online evaluation', 'Separate costly feature preparation from interactive evaluation to improve responsiveness and reduce recurring cost.'],
      ['Multimodal model coverage', 'Black-box and white-box evaluation supports both LLMs and VLMs across different modality-specific tasks.'],
      ['Method comparison and explainable analysis', 'ROC curves, metric tables, and evidence samples make attack results easier to compare and interpret.'],
    ],
  },
}

const STEP_ICONS = [DatabaseOutlined, ScanOutlined, RadarChartOutlined, BarChartOutlined]
const INNOVATION_ICONS = [ApiOutlined, SyncOutlined, PictureOutlined, BarChartOutlined]

export default function ProjectOverviewPanel({ lang, isDark, onEnterConsole }) {
  const text = copy[lang] || copy.zh
  const muted = isDark ? '#a6adc8' : '#64748b'
  const surface = isDark ? '#1e1e2e' : '#ffffff'
  const border = isDark ? '#313244' : '#e5e7eb'

  return (
    <div className="project-overview" style={{ color: isDark ? '#f8fafc' : '#172033' }}>
      <section className="project-hero" style={{ background: surface, borderColor: border }}>
        <div className="project-hero-copy">
          <span className="project-eyebrow">{text.eyebrow}</span>
          <Title level={1} className="project-hero-title">
            {lang === 'zh' ? (
              <>
                让模型记忆风险变得<br />
                <span className="project-hero-emphasis">可测、可见、可验证</span>
              </>
            ) : text.title}
          </Title>
          <Paragraph className="project-hero-lead" style={{ color: muted }}>{text.lead}</Paragraph>
          <Button type="primary" size="large" icon={<ArrowRightOutlined />} iconPosition="end" onClick={onEnterConsole}>
            {text.enter}
          </Button>
        </div>
        <div className="project-signal-map" aria-hidden="true">
          <div className="signal-map-header"><SafetyCertificateOutlined /> Privacy Signal Map</div>
          <div className="signal-map-lines">
            <span className="signal-line signal-line-one" />
            <span className="signal-line signal-line-two" />
            <span className="signal-line signal-line-three" />
          </div>
          <div className="signal-node source-node"><DatabaseOutlined /></div>
          <div className="signal-node model-node"><ApiOutlined /></div>
          <div className="signal-node audit-node"><RadarChartOutlined /></div>
          <div className="signal-caption source-caption">Data</div>
          <div className="signal-caption model-caption">Model</div>
          <div className="signal-caption audit-caption">Audit</div>
          <div className="signal-pill"><span /> Evidence ready</div>
        </div>
      </section>

      <section className="project-section" id="workflow">
        <div className="project-section-heading">
          <span className="project-kicker">01</span>
          <div>
            <Title level={2}>{text.flow}</Title>
            <Paragraph style={{ color: muted }}>{text.flowLead}</Paragraph>
          </div>
        </div>
        <div className="project-flow">
          {text.steps.map(([title, description], index) => {
            const Icon = STEP_ICONS[index]
            return (
              <React.Fragment key={title}>
                <article className="project-flow-step">
                  <div className={`project-step-icon step-tone-${index + 1}`}><Icon /></div>
                  <span className="project-step-index">0{index + 1}</span>
                  <h3>{title}</h3>
                  <p style={{ color: muted }}>{description}</p>
                </article>
                {index < text.steps.length - 1 && <ArrowRightOutlined className="project-flow-arrow" />}
              </React.Fragment>
            )
          })}
        </div>
      </section>

      <section className="project-section project-audience-band" style={{ background: isDark ? '#181825' : '#f8fafc' }}>
        <div className="project-section-heading">
          <span className="project-kicker">02</span>
          <Title level={2}>{text.audience}</Title>
        </div>
        <Row gutter={[28, 28]}>
          <Col xs={24} lg={12}>
            <article className="project-role" style={{ background: surface, borderColor: border }}>
              <div className="project-role-icon role-data"><DatabaseOutlined /></div>
              <div>
                <h3>{text.dataOwner}</h3>
                <p style={{ color: muted }}>{text.dataOwnerDesc}</p>
                <Tag color="cyan">Data governance</Tag>
              </div>
            </article>
          </Col>
          <Col xs={24} lg={12}>
            <article className="project-role" style={{ background: surface, borderColor: border }}>
              <div className="project-role-icon role-model"><TeamOutlined /></div>
              <div>
                <h3>{text.modelOwner}</h3>
                <p style={{ color: muted }}>{text.modelOwnerDesc}</p>
                <Tag color="purple">Privacy self-test</Tag>
              </div>
            </article>
          </Col>
        </Row>
      </section>

      <section className="project-section">
        <div className="project-section-heading">
          <span className="project-kicker">03</span>
          <div>
            <Title level={2}>{text.innovations}</Title>
            <Paragraph style={{ color: muted }}>{text.innovationLead}</Paragraph>
          </div>
        </div>
        <Row gutter={[18, 18]}>
          {text.innovationItems.map(([title, description], index) => {
            const Icon = INNOVATION_ICONS[index]
            return (
              <Col xs={24} sm={12} xl={6} key={title}>
                <article className="project-innovation" style={{ background: surface, borderColor: border }}>
                  <div className={`project-innovation-icon innovation-tone-${index + 1}`}><Icon /></div>
                  <h3>{title}</h3>
                  <p style={{ color: muted }}>{description}</p>
                </article>
              </Col>
            )
          })}
        </Row>
      </section>

      <section className="project-impact" style={{ borderColor: border }}>
        <div>
          <span className="project-kicker">04</span>
          <Title level={2}>{text.impact}</Title>
          <Paragraph style={{ color: muted }}>{text.impactLead}</Paragraph>
        </div>
        <div className="project-impact-points">
          <span><SafetyCertificateOutlined /> Reproducible</span>
          <span><BarChartOutlined /> Comparable</span>
          <span><FileSearchOutlined /> Explainable</span>
        </div>
      </section>

      <section className="project-final-cta" style={{ background: surface, borderColor: border }}>
        <div>
          <Title level={2}>{text.ready}</Title>
          <Paragraph style={{ color: muted }}>{text.readyLead}</Paragraph>
        </div>
        <Button type="primary" size="large" icon={<RadarChartOutlined />} onClick={onEnterConsole}>{text.enter}</Button>
      </section>
    </div>
  )
}
