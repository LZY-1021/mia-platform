import { createContext } from 'react'

export const ThemeContext = createContext({
  isDark: false,
  setIsDark: () => {},
})

export const LangContext = createContext({
  lang: 'zh',
  setLang: () => {},
})