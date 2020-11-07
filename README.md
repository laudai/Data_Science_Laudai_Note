[toc]

# 數學原理
## 線性代數
線性代數式數學的一個分支，處理的是**向量空間**的問題。
### 向量
抽象來說，向量可以彼此相加，以形成一個新向量，也可以乘上一個**純量** (即數字)，以形成令一個新向量。
向量是用來表達數值資料的一種好方法。

以型別別名（type alias）的做法，表達Vector向量，它其實就是一個浮點數列表（如下方code）:
```python
from typing import List
Vector = List[float]
height_weight_age = [70, # 英吋
                    170, # 磅
                    40 # 歲]
```
點積、內積(dot product)。兩個量的點積，就是相應元素相乘之後加總的結果
```python
def dot(v: Vector, w: Vector) -> float:
    """計算v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "兩個向量必須具有相同的維度"
    return sum(v_i * w_i for v_i, w_i in zip(v,w))
assert dot([1, 2, 3], [4, 5, 6]) == 32 # 1 *4 + 2 * 5 + 3 *6
```
引用wiki

> A · B = |A| |B| cos(θ).
> 
> |A| cos(θ)是A到B的投影。
> 也可以說是向量A投影在向量B上的向量長度。
![純量投影圖片](https://upload.wikimedia.org/wikipedia/commons/7/72/Scalarproduct.gif)
※ 範例 code 雖然是用List撰寫，對說明有利但對執行上不利。實際應用會Numpy函式庫，內有高效array陣列物件類別可以使用。


### 矩陣
矩陣指的是一堆二維的數字，通常使用「列表的列表」（list of list）來表示。而矩陣中每行(column)（亦稱作欄）、列(row)都具有相同各數的元素。根據數學上的慣例，我們通常 __使用大寫字母來表示矩陣__。
※ 書中雖然是繁中，但行列翻譯是使用中國習慣的行row、列column。筆者還是習慣從小學習的行column、列row。個人想法，語言就只是敘述某件狀態而已，不須太過糾結，或者使用英文row, column來溝通，個人覺得這樣比較不會有資訊落差。
```python
Martix = List[List[float]]
```
```python
A = [[1, 2, 3],
    [4, 5, 6]]
B = [[1, 2],
    [3, 4],
    [5, 6]]
```
我們可以得知 Martix A.shape (形狀) ＝（2,3），而B.shape = (3,2)(row,column)

`n × k`的矩陣可視為 n列 k行 n row k column的矩陣。


矩陣十分重要：

第一：如果有多個由向量構成的資料集，則可以用矩陣表示。

第二我們可以 __把n × k矩陣當成一個線性函式，把k維的向量映射到n維的向量。__

第三，矩陣可以用來表達二維關係(binary relationship)。

> 個人想法
```
小矩陣還可以用這種二維方式表示，當數量過大時會產生"稀疏矩陣"（sparse matrix）的問題。
這時會產生昂貴的維護成本。
```
作者推薦進階資源：
- 《Linear Algebra》線性代數(http://joshua.smcvt.edu/linearalgebra)
- 《Linear Algebra》線性代數(https://www.math.ucdavis.edu/~linear/linear-guest.pdf)
- 更進階內容 《Linear Algebra Done Wrong》（https://www.math.brown.edu/~treil/papers/LADW/LADW_2017-09-04.pdf）

## 統計學
> 事實往往很棘手，但統計結果看起來好多了。
> 馬克吐溫(_Mark Twain_)
 
我們常用一些統計數字來描述我們的資料，以說明資料本身具有的特性。

### 中央趨勢
**平均值(mean or average)** 也就是把所有資料加起來，再除以資料的數量。
有時會對**中位數(median)** 感興趣，奇數個數中位數為排序後的正中間數值。偶數個數為排序後，中央兩數字平均值。

進階提醒
```
其實有方法可以不必進行資料排序就進行中位數計算，那就是使用Quickselect
```

對平均值資料而言，離群值(outlier或稱作異常值)十分敏感
如果要進一步擴展中位數概念，可以使用「**分位數quantile**」的概念。
**眾數mode**很少用到，是指最常出現或者最多的數值。

### _離散程度_
**離散程度(Dispersion)** 衡量的是資料分散的程度。一般來說，如果這個統計值很接近零，就表示資料分佈很密集。**分散程度很低**;反之，值很大，代表**資料分散得很開**。

範圍range就是其中一種衡量離散程度的最好方式。他是計算資料最大值語最小值得差異。但其實並非是最好呈現方式，e.g. 如果有一組資料不是0就是100，語另外一組大約都在50左右，卻只有一組0, 100，其範圍值相同，但應該是要前一組資料較為分散。

另外一總比較複雜的離散程度度量方式，稱為**變異數(varianace)**，至於為什麼是要n-1，可以參考 https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation。
另外推薦中文樣本變異數介紹：https://highscope.ch.ntu.edu.tw/wordpress/?p=69367 。
當中有提到自由度(degree of freedom) 

資料來源：ntpu網路上ppt的介紹
![image](https://user-images.githubusercontent.com/9117894/97113423-31b36000-1725-11eb-84d9-141540150470.png)

衡量資料離散程度，還有另一種比較可靠的方式，就是計算第75百分位與第25百分位兩個數之間的差值。
e.g. quntile(xs, 0.75) - quntile(xs, 0.25)

### 相關
**共變異數(convariance)**，這個概念語變異數很相似，變異數衡量的是單一變數偏離平均值得程度，**共變異數衡量的是兩個變數分別偏離其平均值的程度**。

案例：要研究使用者在網路上停留的時間，語在網站的朋友數量是否有關係？
透過點積(dot)會把兩兩成對的元素相乘後，再加腫起來的特性。如果相應元素同時大於或小於平均值，相乘就會得到正值，加總就會變大。如果其中有一個比平均值大，另一個比平均值小，相乘就會得到負值，加總就會變小。

但從共變異數無法輕易看出實際上兩個變數的關聯，如當每位使用者的好友數量都增加一倍（但每日上網時間不便）之情況下，共變異數數值就會增加一倍。因此，你需要使用 **_相關係數correlation_** ，其算法為共變異數在各除以雙方的標準差。
> 相關係數本身沒有單位，其數值介於-1（完全負相關）到1（完全正相關）之間。如0.25就表示兩個變數之間存在相對較弱的正相關。

_相關係數容易受到離群值得影響，有興趣可以看書中的程式碼。_

補充資料
```
一般說的相關係數通常是指「皮爾森相關係數(Pearson’s correlation coefficient)」，但當變數之間是順序尺度時用的則是「斯皮爾曼等級相關係數 (Spearman’s rank correlation coefficient)」
其公式算法:p = cov(x,y) / (std(x) * std(y))
```
推薦可以看此文章補強：[相關係數與共變異數(Correlation Coefficient and Covariance)](https://medium.com/@chih.sheng.huang821/%E7%9B%B8%E9%97%9C%E4%BF%82%E6%95%B8%E8%88%87%E5%85%B1%E8%AE%8A%E7%95%B0%E6%95%B8-correlation-coefficient-and-covariance-c9324c5cf679)

個人心得：相關係數來考量兩個變數之間之關聯性，可以很好理解。當 y = x 時，其實公式會從原本的共變異數/std(x)/std(y) 變成=> 樣本變異數x / (標準差x)^2 = 1。因為兩個變數都是相同的，因此相關係數為1，表示完全相關。
### 辛普森悖論
分析資料時，有時會遇到一種叫做辛普森悖論(_Simpson's Pardox_)，它指的是是因為忽略某些變數，造成相關係數被誤導的情況。

現實世界中很長發生這種現象。關鍵在於運用相關係數衡量兩個變數之間的關係時，前提假設是「其他條件完全相同」。如果要避免這種情況，最務實的作法就是更了解資料，盡可能排除所有可能造成混亂的因素。
### 相關係數注意事項
如果相關係數為零，就表示兩個變數之間不存在線性關係。不過，這樣的兩個變數之間，可能還是有其他類型的關係。如：
```python
x = [-2, -1, 0, 1, 2]
y = [ 2,  1, 0, 1, 2]
```
x和y的相關係數為零，但兩個變數之間顯然存在某個關係，因為y的每一個元素，都對應x相應元素的絕對值。

### 相關與因果關係
你可能聽人說過「有相關並不代表因果關係。」有時人麼看到資料所呈現的結論，對自己牢不可破的世界觀形成挑戰時，很有可能會這樣說。

如果x和y呈現強烈的相關性，有可能是因為x影響了y，也有可能因為y影響了x，或是兩者相互影響，**也可能是兩者都被某個第三因素所影響，當然也有可能什麼都不是。**

如果你想進一步掌握因果關係，可以進行隨機試驗。**使用者行為測試**是你可測試的方法之一。

Facebook測試相關文章：[Facebook Tinkers With Users’ Emotions in News Feed Experiment, Stirring Outcry](https://www.nytimes.com/2014/06/30/technology/facebook-tinkers-with-users-emotions-in-news-feed-experiment-stirring-outcry.html)全都廣泛包含各式各樣的統計函式。


進階資料
```
Scipy(https://www.scipy.org/)、pandas(https://pandas.pydata.org/)、statsmodels(https://www.statsmodels.org/stable/index.html)
```
統計學真的很重要，是成為優秀資料科學家必經之路，網路上免費的課程資源：
- 《_Introductory Statistics_》https://open.umn.edu/opentextbooks/textbooks/introductory-statistics
- 《_OnlineStatBook_》https://www.researchgate.net/publication/289555035_Open_Access_Review_of_Online_Statistics_An_Interactive_Multimedia_Course_of_Study_by_David_Lane
- 《_Introductory Statistics_》https://openstax.org/details/introductory-statistics

## 機率
機率就是從事件空間(universe of events)中挑出某些事件(event)，並衡量其中不確定行的一種量化方式。_通常會使用P(E)的寫法，來表示「事件E的機率」_。

### 相依行語獨立性
當發生事件E後，可以/不可以協助判斷事件F是否發生，我們稱之為E和F這兩事件為相依的(dependent)/獨立的(independent)。

數學上角度來看，若兩事件全部發生機率，正好等於兩事件個別發生機率相乘，我們可以說這兩事件是獨立的。
P(E,F)=P(E)P(F)

### 條件機率
如果兩個事件不一定是獨立的（且F的機率不是零），我們即可定義「在F發生的條件下」E發生的機率為：`P(E|F) = P(E,F)/P(F)`

可以用 在以發生F事件，那麼發生E事件的機率是P(E|F)，上面那個式子經常會被重新寫成下面樣子：`P(E,F) = P(E|F)P(F)`

若E和」是獨立的，式子會成為P(E|F) = P (E)

書中是透過`enum.Enum` 列舉出物件，透過迴圈方式去模擬，最後得到的結論近似理論值來實證理論。

### 貝氏定理(Bayes's )
個人認為用這個例子就可以很好舉出該試子如何導出:

∵
```
P(E,F) = P(E|F)P(F)
P(F,E) = P(F|E)P(E)
```
∴
`P(E|F) = P(E,F)/P(F)=P(F|E)P(E)/P(E)`
而P(F) = P(F,E)+P(F,﹁E)
則公式會變成`P(E|F) = P(F|E)P(E)/[P(F|E)P(E)+P(F|﹁E)P(﹁E)]`

以書中的案例而言假設10000人中1人會感染。而正確率為99％
以另外一個直觀思維來看，觀察1,000,000人，真正染病人為100人，但實際上會被偵測出陽性的共99+9999人，而實際染病的只有99人。
換個角度來看：99/(99+9999) = 99/10098 = 0.009803922
驗證了檢測結果為陽性情況下，確實罹患該並機率還不到1%的理論數字。

### 隨機變數
**隨機變數**指的是一種變數，其可能的值與機率分佈有關。
而每個值以其相應機率加權之後所計算出來的平均值，稱之為「期望值」。

### 連續分佈
以丟硬幣的結果而言，對應的是一種「離散分佈（discrete distribution）」，每一個離散的結果都對應到一個正值的機率。

而在「_均勻分佈中(uniform distribution)_」，從0到1之間所有的數值，全都具有相 **_相同的權重_**。

因為0到1之間有無限多個數字，因此，每個數字所分配到的權重也就幾乎趨近於零。我們會用「_機率密度函數(PDF, Probability density function)_」來表示「連續分佈(continuous distribution)」，如此一來，數值落在某個區間的機率，就等於密度函數在該區間的 **_積分_** 結果。

而「_累積分佈函數(CDF, cumulative distribution function)_」，它表示隨機小於等於某個值得機率。
Python程式碼
```python
def uniform_cdf(x: float) -> float:
"""送回均勻分佈隨機變數值 <=x的機率"""
if x < 0:   return 0 # 均勻分佈隨機變數值絕不會小於0
elif x < 1: return x # P(x <=0.4) = 0.4
else:       return 1 # 均勻分佈隨機變數值永遠小於1
```
---
※PDF 個人學習補充
```
在連續變數中，發生在某個數字上的機率為0
連續的東西，關鍵在於密度！
對於隨機變數X而言，其機率密度為如下公式
因此對CDF F_x(x) 微分為 PDF f_x(x)
PDF f_x(x) 積分為 CDF F_x(x)
注記，因為不能在程式段裡使用下標，因此用latex語法 _ 作為下標紀錄

```
![PDF:f_x(x)=\lim_{\Delta x \to 0} \frac {P( x \leq X \leq x+\Delta x)}{ \Delta x}](https://latex.codecogs.com/svg.latex?PDF:f_x(x)=\lim_{\Delta%20x%20\to%200}%20\frac%20{P(%20x%20\leq%20X%20\leq%20x+\Delta%20x)}{%20\Delta%20x})

綜合PDF特性
* f_x(x) = F'_x(x)
* 相反則積分回來即可得到CDF
* 若x位於a b 之間，則機率為fx積分a 到b
* fx從負無窮大到正無窮大，積分為1
* fx 一定大於等於0
* 當x極小時，
* PDF是可以大於一的

※CDF 個人學習補充
* uniform 的cdf 他是連續的，而在某個點上，大家的機率都是相同的
* 下列為CDF公式
![使用wiki的CDF公式圖片](https://wikimedia.org/api/rest_v1/media/math/render/svg/d437ce3920003ecb7c3baa4e3493788c89c8f2be)
* Exponential機率分佈：有失憶的性質(memoryless)，常被用來model有這種性質的事情。
  * e.g.：陷入化妝的小美、陷入無限打電動的阿宅，因為正常而言，會因為一事件花了許多時間，後續會將花在此事的時間減少。
* Erlang機率分佈：Erlang(n,λ)常被用來model一件有多個關卡事情的總時間，而每個關卡所需時間都是隨機的。
  * 舉例：關卡數n
  * 每個關卡所需的機率分佈為Exponential(λ)
  * 打電動過三關所需的時間為Erlang(3,λ)
> 補充<br>
> Erlang 亦稱作Gamma Distribution，其PDF 透過Convolution （中文翻譯褶積）算出

延伸參考資料：
* [台灣大學 葉丙成教授 5-1：機率密度函數 PDF](https://www.coursera.org/lecture/prob2/5-1-ji-lu-mi-du-han-shu-pdf-lTv8h)
* [台灣大學 葉丙成教授 5-2：連續機率分佈 I](https://www.coursera.org/lecture/prob2/5-2-lian-xu-ji-lu-fen-bu-i-XKXWr)


### 常態分佈
常態分佈(normal distribution)是一種典型的鐘形曲線分佈，它可以完緣由兩個參數所決定:平均值μ(念mu)與標準差σ(念sigma)。平均值決定鐘形曲線中央的**位置**，標準差決定曲線的**寬度**。

引用wiki常態分佈公式圖片
![常態分佈圖片參考wiki](https://wikimedia.org/api/rest_v1/media/math/render/svg/21436c871122fb3ba2a676c208afcf91aaebf519)

若μ=0且σ＝1，我們稱之為 **_標準常態分佈_**
如果Z是一個標準常態隨機變數，其常態分佈可用下列表示
`X=σZ+μ`，反之標準常態隨機變數Z 則可用`Z=(X-μ)/σ`算出。

※進階研究
```
常態分佈的CDF累積分佈函式無法使用簡單解析形式表示，有興趣可以參考誤差函式(Error_function)去算出。(Python math.erf)。
若要進行normal_cdf的逆向操作，由特定機率求所對應的值，可以使用二元搜尋(binary search)的方式。
```

### 中央極限定理
中央極限定理：如果每次都從同一個分佈（identically distributed）中獨立(independent)取出大量的隨機變數，然後取齊平均值作為新的隨機變數，那麼這個隨機變數的分佈，_都會趨近於常態分佈_。

當從平均值μ、標準差σ取出隨機n個變數，且n很大。
會得到一個趨近於平均值μ、標準差σ/√n的常態分佈。
而這隨機變數會趨近於一個平均值為0、標準差為1的常態分佈。

亦可通過二項式隨機變數、伯努利隨機變數(Bernoulli)去證實，

自我學習：
```
伯努利分布（英語：Bernoulli distribution，又名兩點分布或者0-1分布
其得到的值不是0就是1，因此值1的機率為p，0的機率為(1-p)，而伯努力變數Bernoulli(p)的平均值為p，標準差則為√p(1-p)。根據中央極限定理，如果n很大，Binoimal(n,p)這個隨機變數會趨近於平均值μ = np、標準差σ=√np(1-p)的常態隨機變數

因此可以把這是為二項式分佈。
證明二項式分佈的影片可以參考
其最後驗證方式會用到「二項式定理」
```
[影片 Binomial Distribution (二項式分配) 期望值證明](http://www.youtube.com/watch?v=SAw2KT4r4YI)<br>
[二項式定理](https://zh.wikipedia.org/zh-tw/%E4%BA%8C%E9%A1%B9%E5%BC%8F%E5%AE%9A%E7%90%86)

進階資料：
```
scipy.stats 包含大部分的常用機率分佈的機率密度函數（PDF）、和累積分佈函數(CDF)
作者推薦網路上最佳統計學書籍： https://math.dartmouth.edu/~prob/prob/prob.pdf
由Grinstead 與 Snell撰寫
```
# 資料視覺化

以matplotlib為例：

install package
```
python -m pip install matplotlib
```

How to use
### 主要關鍵字
```python
from matplotlib import pyplot as plt

# 折線圖
plt.plot(x, y ,color="green", marker="o", linestyle="solid")
# 長條圖
plt.bar(data_len, data)
plt.xticks(range(len(data_str)), data_str) # 設定長條圖x軸標籤
# 折線圖
plt.plot(xs, variance, 'g-',label='varianace')
plt.legend(loc=9) # 設定圖例說明在中間偏上
plt.xticks([]) # 清除下方刻度
plt.show()
# 散點圖
plt.scatter(data_x, data_y)
plt.annotate('your_label', xy=(data_x[0],data_y[0]),xytext=(5,-5),textcoords='offset points')# 在此舉其中一筆資料做標籤，並把該標籤從原位置上稍微位移

```
※ 建議在使用plt.axis時，要特別小心。如果y軸不是從0開始，通常會誤導讀者。
※ plt.axis("equal") 可以讓你在視覺化時，保持間距大小統一。

延伸可學習方向：
* matplotlib Gallery ，讓你知道matplotlib可以做哪些事
* seaborn 是以matplotlib為基礎，進一步構件出來的函式庫，可讓你作出更漂亮（更複雜）的視覺化效果。
* Altair(https://altair-viz.github.io)，較新的Python視覺化函式庫。
* D3.js JavaScript函式庫，雖然不是Python寫的，但它可以製作出更精巧、可在網路上互動的視覺化效果，廣受使用值得你花時間去熟悉它。
* Bokeh 函式庫，具有D3風格視覺化效果帶進入Pyhton世界中。

其它
---
因為突然發現Github 不支援Latex，因此數學是要透過其他方式來呈現，目前暫定使用 https://latex.codecogs.com/ 來建立數學式。
(Latex數學式真的不好寫QQ)
* [常用数学符号的 LaTeX 表示方法](https://www.mohu.org/info/symbols/symbols.htm)

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="創用 CC 授權條款" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />本著作係採用<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">創用 CC 姓名標示-相同方式分享 4.0 國際 授權條款</a>授權.