import React, { useState, useMemo } from 'react'
import { ConfigProvider, theme as antTheme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import enUS from 'antd/locale/en_US'
import DashboardLayout from './layout/DashboardLayout'
import { ThemeContext, LangContext } from './context'

export default function App() {
  const [isDark, setIsDark] = useState(false)
  const [lang, setLang] = useState('zh')

  const themeCtx = useMemo(() => ({ isDark, setIsDark }), [isDark])
  const langCtx = useMemo(() => ({ lang, setLang }), [lang])

  return (
    <ThemeContext.Provider value={themeCtx}>
      <LangContext.Provider value={langCtx}>
        <ConfigProvider
          locale={lang === 'zh' ? zhCN : enUS}
          theme={{
            algorithm: isDark ? antTheme.darkAlgorithm : antTheme.defaultAlgorithm,
            token: {
              colorPrimary: '#4f46e5',
              borderRadius: 8,
              colorBgContainer: isDark ? '#1e1e2e' : '#ffffff',
              colorBgLayout: isDark ? '#11111b' : '#f5f5f5',
            },
          }}
        >
          <DashboardLayout />
        </ConfigProvider>
      </LangContext.Provider>
    </ThemeContext.Provider>
  )
}
