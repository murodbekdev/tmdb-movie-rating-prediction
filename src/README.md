# 🎬 TMDB Movies Dataset

## 📌 Overview

**TMDB Movies Dataset** — filmlar haqida turli xil ma'lumotlarni o'z ichiga olgan dataset bo'lib, ma'lumotlar **The Movie Database (TMDB)** platformasidan yig'ilgan.

Dataset film metadata'lari, reytinglar, moliyaviy ko'rsatkichlar, janrlar, ijodkorlar va boshqa muhim atributlarni o'z ichiga oladi.

Ushbu dataset asosan **Data Analysis, Exploratory Data Analysis (EDA), Data Preprocessing, Feature Engineering va Machine Learning** loyihalarida foydalanish uchun mo'ljallangan.

Dataset yordamida filmlarning reytinglariga ta'sir qiluvchi omillarni o'rganish, filmlarni tahlil qilish va turli xil Machine Learning modellarini qurish mumkin.

---

## 📊 Features

Dataset quyidagi asosiy ustunlardan tashkil topgan:

| Column | Description |
|---|---|
| `movie_id` | TMDB platformasidagi filmning noyob identifikatori. |
| `url` | Filmning TMDB platformasidagi sahifa manzili. |
| `title` | Film nomi. |
| `runtime` | Film davomiyligi, daqiqalarda. |
| `release_date` | Filmning chiqarilgan sanasi. |
| `release_country` | Film chiqarilgan yoki ma'lumot olingan mamlakat kodi. |
| `genres` | Filmga tegishli janrlar, masalan Drama, Comedy, Action va boshqalar. |
| `user_score` | TMDB foydalanuvchilari tomonidan berilgan film reytingi. |
| `overview` | Filmning qisqacha mazmuni yoki syujeti. |
| `director` | Film rejissyori. |
| `writer` | Film ssenariysini yozgan muallif yoki mualliflar. |
| `status` | Filmning ishlab chiqarish/chiqarilish holati. |
| `original_language` | Filmning original tili. |
| `budget` | Filmni ishlab chiqarish uchun sarflangan taxminiy budjet. |
| `revenue` | Filmning taxminiy umumiy daromadi. |
| `tmdb_movie` | TMDB ma'lumotlaridan foydalanilgan yoki qayta ishlangan datasetdagi qo'shimcha identifikatsiya/klassifikatsiya qiymati. |

---

## 🔍 Data Collection

Dataset **The Movie Database (TMDB)** platformasidagi ochiq film ma'lumotlari asosida yig'ilgan.

Ma'lumotlarni yig'ish jarayonida Python dasturlash tilidan va web scraping texnologiyalaridan foydalanilgan.

Asosiy texnologiyalar:

- Python
- `requests`
- `BeautifulSoup`
- `pandas`

Data collection jarayoni umumiy tarzda quyidagi bosqichlardan tashkil topgan:

1. TMDB film sahifalariga HTTP request yuborish.
2. Film sahifalaridagi kerakli ma'lumotlarni olish.
3. HTML ma'lumotlarini `BeautifulSoup` yordamida parse qilish.
4. Kerakli atributlarni ajratib olish.
5. Ma'lumotlarni `pandas DataFrame` formatiga o'tkazish.
6. Yig'ilgan ma'lumotlarni CSV formatida saqlash.

Datasetda film nomi, davomiyligi, chiqarilgan sana, janr, reyting, rejissyor, ssenariynavis, budjet, daromad va boshqa atributlar jamlangan.

> **Note:** Ba'zi filmlar uchun ayrim ma'lumotlar TMDB sahifasida mavjud bo'lmasligi sababli `missing values` mavjud bo'lishi mumkin.

---

## 🧹 Data Cleaning

Dataset Machine Learning va Data Analysis uchun ishlatilishidan oldin bir nechta preprocessing bosqichlaridan o'tkazilgan.

Asosiy cleaning jarayonlari:

- Missing valueslarni aniqlash.
- Numerical va categorical ustunlarni ajratish.
- Noto'g'ri yoki yetishmayotgan qiymatlarni tekshirish.
- Duplicate ma'lumotlarni tekshirish.
- Date formatlarini standartlashtirish.
- Categorical ma'lumotlarni encoding qilish.
- Numerical features uchun scaling.
- Skewed distributionslarni tekshirish va kerak bo'lganda transformatsiya qilish.

---

## ⚙️ Feature Engineering

Datasetdan Machine Learning modellari uchun qo'shimcha featurelar yaratish mumkin.

Ushbu loyiha davomida quyidagi derived featureslardan foydalanilgan:

| Feature | Description |
|---|---|
| `release_year` | `release_date` ustunidan film chiqarilgan yilni ajratib olish. |
| `release_month` | Film chiqarilgan oyni ajratib olish. |
| `movie_age` | Film chiqarilgan yildan kelib chiqib film yoshini hisoblash. |
| `genre_count` | Filmga tegishli janrlar soni. |

Feature Engineering jarayoni modelga foydali bo'lishi mumkin bo'lgan yangi ma'lumotlarni mavjud ustunlardan hosil qilishga yordam beradi.

---

## ❓ Missing Values

TMDB ma'lumotlarida ayrim ustunlarda missing values mavjud bo'lishi mumkin.

Masalan:

- `runtime`
- `genres`
- `director`
- `writer`
- `budget`
- `revenue`

Missing valueslarni boshqarish uchun quyidagi usullardan foydalanish mumkin:

- Mean Imputation
- Median Imputation
- KNN Imputation
- Iterative Imputation
- MICE

Categorical ustunlar uchun esa mos categorical imputation usullaridan foydalanish mumkin.

Missing valuesni to'g'ri boshqarish Machine Learning modelining barqarorligi va natijalariga ta'sir qiladi.

---

## 🤖 Possible Applications / Use Cases

Ushbu dataset turli xil Data Science va Machine Learning vazifalarida foydalanilishi mumkin.

### 1. 🎯 Movie Rating Prediction

Filmning:

- janri,
- davomiyligi,
- budjeti,
- daromadi,
- chiqarilgan yili,
- rejissyori

kabi xususiyatlaridan foydalanib `user_score` qiymatini bashorat qilish mumkin.

### 2. 📈 Exploratory Data Analysis

Filmlar haqida quyidagi savollarni o'rganish mumkin:

- Qaysi janrlar eng ko'p uchraydi?
- Film davomiyligi reytingga ta'sir qiladimi?
- Budjet va revenue o'rtasida qanday bog'liqlik mavjud?
- Qaysi yillarda ko'proq filmlar chiqarilgan?
- Film yoshi va reyting o'rtasida qanday munosabat mavjud?

### 3. 💰 Revenue Analysis

Film budjeti va daromadi o'rtasidagi munosabatni o'rganish mumkin.

### 4. 🎬 Genre Analysis

Turli janrlarning:

- reytinglari,
- soni,
- budjeti,
- daromadi

bo'yicha taqqoslashlar o'tkazish mumkin.

### 5. 🤖 Machine Learning

Dataset quyidagi Machine Learning algoritmlarini sinash uchun ishlatilishi mumkin:

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine
- K-Nearest Neighbors

### 6. 📊 Feature Selection

Correlation analysis, variance analysis va model-based feature selection yordamida eng muhim featurelarni aniqlash mumkin.

---

## ⚠️ Limitations

Datasetdan foydalanishda quyidagi cheklovlarni hisobga olish kerak:

- TMDB'dagi ayrim filmlar uchun ma'lumotlar to'liq bo'lmasligi mumkin.
- Ba'zi filmlarda `budget` yoki `revenue` qiymatlari mavjud bo'lmasligi mumkin.
- Categorical ustunlarda juda ko'p unique qiymatlar mavjud bo'lishi mumkin.
- Web scraping jarayonida ayrim sahifalardan ma'lumot olish muvaffaqiyatsiz bo'lishi mumkin.
- `user_score` vaqt o'tishi bilan o'zgarishi mumkin.
- Dataset TMDB platformasidagi mavjud ma'lumotlarga bog'liq.
- Datasetdagi ma'lumotlar real hayotdagi barcha film xususiyatlarini to'liq ifodalamasligi mumkin.

---

## 📜 License & Source

### Data Source

Ma'lumotlarning asosiy manbasi:

**The Movie Database (TMDB)**

Dataset TMDB platformasidagi film ma'lumotlari asosida yaratilgan.

> This dataset is an independent collection of movie metadata obtained from publicly available information on TMDB. It is not an official TMDB dataset.

TMDB haqida qo'shimcha ma'lumot:

https://www.themoviedb.org/

### License

Ushbu datasetning qayta tarqatilishi va tijoriy foydalanish shartlari original TMDB ma'lumotlarining tegishli litsenziya va foydalanish shartlariga bog'liq.

Datasetdan foydalanishda TMDB platformasining **Terms of Use** va **API/attribution requirements** talablarini tekshirish tavsiya etiladi.

---

## 🏷️ Tags

`tmdb`  
`movies`  
`movie-dataset`  
`data-science`  
`machine-learning`  
`data-analysis`  
`eda`  
`feature-engineering`  
`regression`  
`python`  
`pandas`  
`web-scraping`  
`data-preprocessing`  
`movie-rating-prediction`

---

## 🛠️ Technologies

Ushbu dataset va loyiha bilan ishlashda quyidagi texnologiyalardan foydalanish mumkin:

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **BeautifulSoup**
- **Requests**
- **Matplotlib**
- **Plotly**
- **XGBoost**
- **Joblib**

---

## 📁 Dataset Structure

```text
TMDB Movies Dataset
│
├── movie_id
├── url
├── title
├── runtime
├── release_date
├── release_country
├── genres
├── user_score
├── overview
├── director
├── writer
├── status
├── original_language
├── budget
├── revenue
└── tmdb_movie