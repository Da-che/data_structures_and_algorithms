# Assignment #3: 惊蛰 Mock Exam

Updated 1641 GMT+8 Mar 5, 2025

2025 spring, Complied by 陈宣之 生命科学学院 <mark>同学的姓名、院系</mark>



> **说明：**
>
> 1. **惊蛰⽉考**：AC4<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E04015: 邮箱验证

strings, http://cs101.openjudge.cn/practice/04015



思路：排除首位与相连的情况，然后扫描计数@，判断之后出现.并且计数为1



代码：

```python
while True:
    try:
        mail=input()
        if mail[0]=="@" or mail[-1]=="@" or mail[0]=="." or mail[-1]=="." or ".@" in mail or "@." in mail:
            print("NO")
            continue
        cnt=0
        judge=False
        for i in mail:
            if i=="@":
                cnt+=1
            if i=="." and cnt==1:
                judge=True
        if cnt==1 and judge==True:
            print("YES")
        else:
            print("NO")

    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250310225832653](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250310225832653.png)



### M02039: 反反复复

implementation, http://cs101.openjudge.cn/practice/02039/



思路：point指针，dire设置方向存入matrix，再竖着输出



代码：

```python
m=int(input())
info=input()
matrix=[]
point=m
dire=1
while point<=len(info):
    if dire==1:
        matrix.append(info[point-m:point])
    else:
        matrix.append(info[point-1:point-m-1:-1])
    point+=m
    dire*=(-1)
for i in range(m):
    for j in range(len(matrix)):
        print(matrix[j][i],end="")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250310230020587](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250310230020587.png)



### M02092: Grandpa is Famous

implementation, http://cs101.openjudge.cn/practice/02092/



思路：用defaltdict，键为身份序号，值计数，按值倒序排列items，找到第二的值no2，再找值为no2的键，正序输出



代码：

```python
from collections import defaultdict
while True:
    n,m=map(int,input().split())
    if n==0 and m==0:
        break
    else:
        rank=defaultdict(int)
        for i in range(n):
            lst=list(map(int,input().split()))
            for i in lst:
                rank[i]+=1
        rank=sorted(rank.items(),key=lambda x:x[1],reverse=True)
        no1,no2=rank[0][1],0
        ans=[]
        for i in rank:
            if i[1]<no1:
                no2=i[1]
                break
        for i in rank:
            if i[1]==no2:
                ans.append(i[0])
            if i[1]<no2:
                break
        ans.sort()
        print(*ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250310230422886](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250310230422886.png)



### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/



思路：遍历每个点维护最大值



代码：

```python
d=int(input())
n=int(input())
rub=[]
for i in range(n):
    rub.append(list(map(int,input().split())))
ans=0
cnt=0
for i in range(1025):
    for j in range(1025):
        temp=0
        for scan in rub:
            if abs(scan[0]-i)<=d and abs(scan[1]-j)<=d:
                temp+=scan[2]
        if temp>ans:
            ans=temp
            cnt=1
        elif temp==ans:
            cnt+=1
print(cnt,ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250310230517260](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250310230517260.png)



### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/



思路：Dijstra，把路径放在第一位，用heapq取字典序最小



代码：

```python
import heapq
def dfs(x,y,r,c):
    global table,directions
    q=[[table[x][y],(x,y)]]
    while q:
        way,(x,y)=heapq.heappop(q)
        if len(way)==r*c*2:
            return way
        for dx,dy in directions:
            nx,ny=x+dx,y+dy
            if 0<=nx<r and 0<=ny<c and table[nx][ny] not in way:
                heapq.heappush(q,[way+table[nx][ny],(nx,ny)])
    return 0


n=int(input())
directions=[(-2,-1),(-2,1),(-1,2),(-1,-2),(1,-2),(1,2),(2,-1),(2,1)]
for _ in range(n):
    p,q=map(int,input().split())
    table=[]
    for i in range(p):
        temp=[]
        for j in range(q):
            temp.append(chr(ord("A")+j)+str(i+1))
        table.append(temp)
    judge=False
    for j in range(q):
        for i in range(p):
            if dfs(i,j,p,q):
                judge=True
                print("Scenario #",_+1,":",sep="")
                print(dfs(i,j,p,q))
                break
        if judge:
            break
    if not judge:
        print("Scenario #",_+1,":",sep="")
        print("impossible")
    if _<n-1:
        print()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250310230752263](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250310230752263.png)



### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/



思路：输入时将各条序列sort，先只考虑两条序列，（0,0）一定最小，用heapq存储，下一步最小一定在（i+1, j）和（i, j+1）之间，以此类推找到最小的n个存为序列seq，再将seq与第三条序列重复操作，以此类推。注意m=1的情况。



代码：

```python
import heapq
def step(seq1,seq2):
    global n,m
    res=[]
    q=[(seq1[0]+seq2[0],0,0)]
    inq=set((0,0))
    while q and len(res)<n:
        min_sum,seq1_idx,seq2_idx=heapq.heappop(q)
        res.append(min_sum)
        if seq1_idx<n-1 and (seq1_idx+1,seq2_idx) not in inq:
            heapq.heappush(q,(seq1[seq1_idx+1]+seq2[seq2_idx],seq1_idx+1,seq2_idx))
            inq.add((seq1_idx+1,seq2_idx))
        if seq2_idx<n-1 and (seq1_idx,seq2_idx+1) not in inq:
            heapq.heappush(q,(seq1[seq1_idx]+seq2[seq2_idx+1],seq1_idx,seq2_idx+1))
            inq.add((seq1_idx,seq2_idx+1))
    return res

t=int(input())
for i in range(t):
    m,n=map(int,input().split())
    sequences=[]
    for j in range(m):
        temp=list(map(int,input().split()))
        temp.sort()
        sequences.append(temp)
    if len(sequences)<2:
        print(*sequences[0])
    else:
        seq=step(sequences[0],sequences[1])
        for j in range(2,m):
            seq=step(seq,sequences[j])
        print(*seq)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20250311222746535](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250311222746535.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

月考AC4，第五题做出来了但没来得及提交（这种情况居然真的发生了qwq），最后一题比较难，考试结束后研究了挺久的，大概知道要用heapq，复杂度压到n*m量级，但是总把m条放在一起想，导致step+1操作时的可能就不止m种了，保证最小就要回溯，那就变成遍历了，复杂度肯定要超，吃饭的时候看看微信群，知道了两条两条找就没有这个麻烦了，感觉很巧妙，从局部最优到整体最优，回去很快就解决了。
