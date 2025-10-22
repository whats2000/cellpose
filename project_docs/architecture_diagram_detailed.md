# 骨髓幹細胞年齡分析：詳細架構圖集

## 1. 整體系統架構（系統視角）

```mermaid
graph TB
    subgraph Input["輸入資料層"]
        A1["少量標記資料<br/>~20 images<br/>Roboflow COCO"]
        A2["大量未標記資料<br/>Adult & Pediatric MSC<br/>全視野切片"]
    end
    
    subgraph Processing["處理與訓練層"]
        B1["影像預處理<br/>Tiling & Normalization"]
        B2["Cellpose-SAM<br/>分割模型"]
        B3["細胞分類器<br/>早/中/晚期"]
        B4["主動學習引擎<br/>Uncertainty + Diversity"]
    end
    
    subgraph Human["人機協同層"]
        C1["樣本選擇<br/>50-200/iteration"]
        C2["Cellpose GUI<br/>標註介面"]
        C3["品質檢查<br/>QC & Validation"]
    end
    
    subgraph Training["迭代訓練層"]
        D1["增量訓練"]
        D2["模型驗證"]
        D3{達標?}
        D4["下一輪迭代"]
    end
    
    subgraph Output["輸出分析層"]
        E1["族群統計"]
        E2["年齡分佈"]
        E3["視覺化報告"]
    end
    
    A1 --> B1
    A2 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> D1
    D1 --> D2
    D2 --> D3
    D3 -->|否| D4
    D4 --> B4
    D3 -->|是| E1
    E1 --> E2
    E2 --> E3
    
    style Input fill:#e3f2fd
    style Processing fill:#fff3e0
    style Human fill:#ffebee
    style Training fill:#f3e5f5
    style Output fill:#e8f5e9
```

---

## 2. 主動學習迭代流程（時間序列視角）

```mermaid
gantt
    title 主動學習迭代時程表
    dateFormat YYYY-MM-DD
    section 準備階段
    環境建置           :2025-10-16, 5d
    資料整理           :2025-10-18, 4d
    section 初始訓練
    基礎模型訓練       :2025-10-23, 7d
    效能評估           :2025-10-30, 2d
    section 迭代1
    樣本選擇           :2025-11-01, 1d
    人工標註           :2025-11-02, 3d
    增量訓練           :2025-11-05, 2d
    驗證評估           :2025-11-07, 1d
    section 迭代2-8
    持續迭代           :2025-11-08, 56d
    section 最終部署
    大規模預測         :2026-01-03, 10d
    統計分析           :2026-01-13, 5d
    報告撰寫           :2026-01-18, 7d
```

---

## 3. 資料流轉圖（資料視角）

```mermaid
flowchart LR
    subgraph Raw["原始資料"]
        R1[("Roboflow<br/>標記資料")]
        R2[("IX83 未標記<br/>Adult MSC")]
        R3[("IX83 未標記<br/>Pediatric MSC")]
    end
    
    subgraph Prep["預處理"]
        P1["切片<br/>Tiling"]
        P2["標準化<br/>Normalization"]
        P3["增強<br/>Augmentation"]
    end
    
    subgraph Pool["資料池"]
        D1[("訓練集<br/>Training")]
        D2[("驗證集<br/>Validation")]
        D3[("待標註池<br/>Unlabeled Pool")]
    end
    
    subgraph AL["主動學習"]
        A1["預測<br/>Prediction"]
        A2["不確定性<br/>Uncertainty"]
        A3["選擇樣本<br/>Selection"]
    end
    
    subgraph Anno["標註"]
        N1["人工標註<br/>Annotation"]
        N2["品質檢查<br/>QC"]
    end
    
    R1 --> P2
    R2 --> P1
    R3 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> D1
    P3 --> D2
    P2 --> D3
    
    D3 --> A1
    A1 --> A2
    A2 --> A3
    A3 --> N1
    N1 --> N2
    N2 --> D1
    
    D1 -.下一輪.-> A1
    
    style Raw fill:#e1f5fe
    style Prep fill:#fff9c4
    style Pool fill:#f3e5f5
    style AL fill:#ffe0b2
    style Anno fill:#ffcdd2
```

---

## 4. 不確定性計算流程（演算法視角）

```mermaid
flowchart TB
    A["輸入影像<br/>Input Image"] --> B["Cellpose 分割<br/>Segmentation"]
    B --> C["提取細胞<br/>Cell Extraction"]
    C --> D["特徵向量<br/>Feature Vector"]
    
    D --> E1["分類器預測<br/>Classifier Prediction"]
    D --> E2["MC Dropout × 10<br/>Multiple Predictions"]
    
    E1 --> F1["機率分佈<br/>P(early, middle, late)"]
    E2 --> F2["預測變異<br/>Prediction Variance"]
    
    F1 --> G1["預測熵<br/>Entropy = -Σp·log(p)"]
    F1 --> G2["邊界距離<br/>Margin = p₁ - p₂"]
    F2 --> G3["變異係數<br/>Variance Ratio"]
    
    G1 --> H["不確定性分數<br/>Uncertainty Score"]
    G2 --> H
    G3 --> H
    
    H --> I["排序與選擇<br/>Ranking & Selection"]
    I --> J["前 N 個樣本<br/>Top-N Samples"]
    J --> K["人工標註<br/>Human Annotation"]
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style D fill:#f3e5f5
    style H fill:#ffcdd2
    style K fill:#c8e6c9
```

---

## 5. 模型訓練策略（技術視角）

```mermaid
flowchart TB
    subgraph Init["初始階段 Iteration 0"]
        I1["Cellpose-SAM<br/>預訓練模型"]
        I2["Roboflow 資料<br/>~20 images"]
        I3["微調訓練<br/>Fine-tuning"]
        I4["基礎分類器<br/>3-class Classifier"]
    end
    
    subgraph Iter["迭代階段 Iteration 1-N"]
        L1["選擇樣本<br/>Select 50-200"]
        L2["人工標註<br/>Annotate"]
        L3["合併資料<br/>Merge Data"]
        L4["增量訓練<br/>Incremental Training"]
        L5["效能驗證<br/>Validation"]
        L6{效能提升?}
    end
    
    subgraph Deploy["部署階段"]
        D1["最佳模型<br/>Best Model"]
        D2["批次預測<br/>Batch Inference"]
        D3["後處理<br/>Post-processing"]
        D4["統計分析<br/>Statistics"]
    end
    
    I1 --> I3
    I2 --> I3
    I3 --> I4
    I4 --> L1
    
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
    L5 --> L6
    
    L6 -->|是| L1
    L6 -->|否 or 達標| D1
    
    D1 --> D2
    D2 --> D3
    D3 --> D4
    
    style Init fill:#e8eaf6
    style Iter fill:#fff3e0
    style Deploy fill:#e8f5e9
```

---

## 6. 細胞分類決策樹（生物學視角）

```mermaid
graph TD
    A[細胞實例<br/>Cell Instance] --> B{分割品質}
    B -->|差| C[排除<br/>Exclude]
    B -->|好| D[特徵提取<br/>Feature Extraction]
    
    D --> E1[形態特徵<br/>Morphology]
    D --> E2[紋理特徵<br/>Texture]
    D --> E3[強度特徵<br/>Intensity]
    
    E1 --> F{分類器<br/>Classifier}
    E2 --> F
    E3 --> F
    
    F --> G1[早期<br/>Early Stage]
    F --> G2[中期<br/>Middle Stage]
    F --> G3[晚期<br/>Late Stage]
    
    G1 --> H{信心度}
    G2 --> H
    G3 --> H
    
    H -->|高| I[自動接受<br/>Auto Accept]
    H -->|低| J[人工審核<br/>Human Review]
    
    style A fill:#e3f2fd
    style D fill:#fff9c4
    style F fill:#f3e5f5
    style G1 fill:#c8e6c9
    style G2 fill:#fff9c4
    style G3 fill:#ffcdd2
```

---

## 7. 人機介面互動流程（使用者視角）

```mermaid
sequenceDiagram
    autonumber
    participant S as 主動學習系統
    participant G as Cellpose GUI
    participant U as 標註人員
    participant M as 模型引擎
    participant D as 資料庫
    
    S->>S: 計算不確定性分數
    S->>S: 選擇 Top-N 樣本
    S->>G: 載入待標註影像
    
    loop 每張影像
        G->>U: 顯示影像 + 模型預測
        U->>G: 修正分割輪廓
        U->>G: 標記細胞類別（早/中/晚）
        U->>G: 標記品質標籤
        G->>D: 儲存標註結果
    end
    
    D->>S: 回傳新標註資料
    S->>M: 觸發增量訓練
    
    M->>M: 載入前一版模型
    M->>M: 合併新舊資料
    M->>M: Fine-tuning 50 epochs
    M->>M: 驗證效能
    
    M->>S: 回傳新模型
    S->>S: 評估效能指標
    
    alt 效能達標
        S->>U: 通知完成訓練
    else 效能未達標
        S->>S: 準備下一輪迭代
    end
```

---

## 8. 效能監控儀表板（管理視角）

```mermaid
graph TB
    subgraph Metrics["效能指標 Performance Metrics"]
        M1["分割 IoU<br/>Target: >0.75"]
        M2["分類準確率<br/>Target: >0.90"]
        M3["F1-Score<br/>Target: >0.85"]
    end
    
    subgraph Progress["訓練進度 Training Progress"]
        P1["迭代次數<br/>Current: 3/10"]
        P2["已標註數量<br/>320 images"]
        P3["待標註池<br/>~5000 images"]
    end
    
    subgraph Quality["品質控制 Quality Control"]
        Q1["標註一致性<br/>Inter-annotator Agreement"]
        Q2["模型信心度分佈<br/>Confidence Distribution"]
        Q3["錯誤案例分析<br/>Error Analysis"]
    end
    
    subgraph Resource["資源使用 Resource Usage"]
        R1["GPU 使用率<br/>75%"]
        R2["訓練時間<br/>2.5 hrs/iteration"]
        R3["標註時間<br/>12 hrs/iteration"]
    end
    
    M1 -.監控.-> P1
    M2 -.監控.-> P1
    M3 -.監控.-> P1
    
    P2 -.影響.-> M2
    P3 -.影響.-> P1
    
    Q1 -.影響.-> M2
    Q2 -.影響.-> P1
    
    style Metrics fill:#e8f5e9
    style Progress fill:#e3f2fd
    style Quality fill:#fff3e0
    style Resource fill:#f3e5f5
```

---

## 9. 風險緩解策略（風險管理視角）

```mermaid
mindmap
  root((風險管理))
    資料風險
      類別不平衡
        加權損失函數
        SMOTE 過採樣
        Focal Loss
      影像品質差異
        標準化流程
        影像增強
        多模態訓練
    模型風險
      過擬合
        Dropout
        Data Augmentation
        Early Stopping
      收斂緩慢
        學習率調整
        遷移學習
        更多標註
    人力風險
      標註不一致
        標註指引手冊
        定期校準會議
        多人標註驗證
      人力不足
        優化標註介面
        半自動標註
        外包標註
    資源風險
      GPU 資源限制
        雲端 GPU
        模型量化
        批次大小調整
      儲存空間不足
        資料壓縮
        定期清理
        雲端儲存
```

---

## 10. 成功路徑圖（目標視角）

```mermaid
journey
    title 主動學習成功之路
    section 第1個月
      環境建置: 5: Me, Team
      資料準備: 4: Me, Team
      初始訓練: 3: Me
      迭代1-2: 4: Me, Annotators
    section 第2個月
      迭代3-5: 5: Me, Annotators
      模型優化: 4: Me
      中期評估: 5: Me, Team
      迭代6-8: 4: Me, Annotators
    section 第3個月
      最終迭代: 5: Me, Annotators
      大規模預測: 4: Me
      品質檢查: 5: Me, Team
      統計分析: 5: Me, Team
    section 第4個月
      結果視覺化: 5: Me
      報告撰寫: 5: Me, Team
      論文發表: 5: Team
```

---

## 使用說明

### 圖表用途
1. **圖1-2**：用於整體架構說明與時程規劃
2. **圖3-5**：用於技術細節說明
3. **圖6-7**：用於演示操作流程
4. **圖8-10**：用於專案管理與風險控制

### Mermaid 渲染
這些圖表使用 Mermaid 語法，可在以下平台渲染：
- GitHub / GitLab（自動渲染）
- VS Code（安裝 Mermaid 插件）
- [Mermaid Live Editor](https://mermaid.live/)
- Notion、Obsidian 等筆記軟體

### 匯出建議
- **簡報用**：匯出為 PNG/SVG 高解析度圖片
- **報告用**：直接嵌入 Markdown 文件
- **網頁用**：使用 Mermaid.js 動態渲染

---

**文件版本**：1.0  
**建立日期**：2025-10-16  
**適用專案**：骨髓幹細胞年齡分析
