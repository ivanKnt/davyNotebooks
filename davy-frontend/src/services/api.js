import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
})

// Preprocessing
export const runPreprocessing = () => api.post('/preprocessing/run')
export const getPreprocessingStatus = () => api.get('/preprocessing/status')

// Classification
export const runClassification = () => api.post('/classification/run')
export const listClassificationNotebooks = () => api.get('/classification/notebooks')
export const getNotebookClassification = (id) => api.get(`/classification/notebook/${id}`)
export const getPageClassification = (id, page) => api.get(`/classification/page/${id}/${page}`)

// Poetry
export const runPoetry = () => api.post('/poetry/run')
export const listPoetryNotebooks = () => api.get('/poetry/notebooks')
export const listPoetryPages = () => api.get('/poetry/pages')
export const listPoetryPagesForNotebook = (id) => api.get(`/poetry/pages/${id}`)

// Text Reuse
export const listTRNotebooks = () => api.get('/text-reuse/notebooks')
export const listTRConfigs = (alg) => api.get(`/text-reuse/configs/${alg}`)
export const runTextReuse = (payload) => api.post('/text-reuse/run', payload)
export const getTextReuseResults = (alg, configId, notebooks) => api.get(`/text-reuse/results/${alg}/${configId}/${notebooks}`)
export const getTextReuseStatus = (token) => api.get(`/text-reuse/status/${token}`)


