import axios from 'axios'

const API = axios.create({
  baseURL: '/api',
  timeout: 300000,
})

export async function fetchAttacks() {
  const res = await API.get('/attacks')
  return res.data
}

export async function fetchParams(attackId) {
  const res = await API.get(`/attacks/${attackId}/params`)
  return res.data
}

export async function fetchData(attackId) {
  const res = await API.get(`/attacks/${attackId}/data`)
  return res.data
}

export async function runAttack(attackId, dataPath, params, useCache = true) {
  const res = await API.post(`/attacks/${attackId}/run`, {
    data_path: dataPath,
    params,
    use_cache: useCache,
  })
  return res.data
}

export async function fetchRunLogs() {
  const res = await API.get('/runs')
  return res.data.runs || []
}

export async function fetchRunLog(runId) {
  const res = await API.get(`/runs/${runId}`)
  return res.data
}
