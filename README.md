# LINE BOT 智慧防詐平台

這是一個結合生成式 AI 技術的 LINE 機器人，專為識別與防範詐騙行為而設計。透過整合 GPT-3.5-turbo 與 RAG（檢索增強生成）技術，系統能即時分析使用者輸入的訊息或語音，並對比專業的防詐騙資料庫，提供精確的防範資訊與建議。

## 🌟 核心功能

*   **即時防詐諮詢**：透過 LINE 介面與 AI 進行對話，詢問可疑訊息是否為詐騙。
*   **RAG 知識檢索**：結合 LangChain 與 FAISS 向量資料庫，從預先載入的防詐騙 PDF 文件中檢索最相關的防範資訊。
*   **語音識別分析**：整合 OpenAI Whisper 模型，支援接收語音訊息並轉換為文字進行分析，方便不便打字的使用者。
*   **對話記錄管理**：具備記憶功能，能根據上下文提供更精確的建議；支援 `/clear` 指令重置對話記錄。

## 🛠️ 技術架構

*   **後端框架**：[Flask](https://flask.palletsprojects.com/) (Python)
*   **AI 模型**：
    *   **LLM**: OpenAI `gpt-3.5-turbo`
    *   **語音轉文字**: OpenAI `Whisper-1`
    *   **文本嵌入**: `OpenAIEmbeddings`
*   **RAG 框架**：[LangChain](https://www.langchain.com/)
*   **向量資料庫**：[FAISS](https://github.com/facebookresearch/faiss)
*   **LINE SDK**：`line-bot-sdk`


## 📂 專案結構

*   [app.py](cci:7://file:///d:/chat_bot/app.py:0:0-0:0): 專案主程式，處理 LINE Webhook 請求、訊息邏輯及語音轉換。
*   [src/pdf.py](cci:7://file:///d:/chat_bot/src/pdf.py:0:0-0:0): 處理 PDF 文件讀取、文本分割（RecursiveCharacterTextSplitter）及向量資料庫初始化。
*   `06afterclean/`: 存放防詐騙宣導或相關資訊的 PDF 資料夾。
*   `faiss_midjourney_docs/`: 本地儲存的 FAISS 向量索引。
*   [requirements.txt](cci:7://file:///d:/chat_bot/requirements.txt:0:0-0:0): 專案依賴套件清單。

## 🚀 快速上手

### 1. 環境設定
請在專案根目錄建立 `.env` 檔案，填入以下金鑰：
```env
LINE_ACCESS_TOKEN=你的_LINE_Channel_Access_Token
LINE_SECRET=你的_LINE_Channel_Secret
OPENAI_API_KEY=你的_OpenAI_API_Key
