# Assignment #6: 回溯、树、双向链表和哈希表

Updated 1526 GMT+8 Mar 22, 2025

2025 spring, Complied by <mark>陈宣之 生命科学学院</mark>



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### LC46.全排列

backtracking, https://leetcode.cn/problems/permutations/

思路：

dfs搜索，used存nums各位是否已在当前align，若为False则加入align，used设为True继续调用函数递推，然后回溯还原状态，当align长度为n时加入ans输出（注意深拷贝）

代码：

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n=len(nums)
        ans=[]
        used=[False]*n
        def permutaion(align=[]):
            if len(align)==n:
                ans.append(align[:])
                return
            for i in range(n):
                if used[i]==False:
                    align.append(nums[i])
                    used[i]=True
                    permutaion(align)
                    used[i]=False
                    align.pop()
            return ans
        permutaion([])
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401094601511](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250401094601511.png)



### LC79: 单词搜索

backtracking, https://leetcode.cn/problems/word-search/

思路：

dfs搜索，visited存储已经过的位置，四个方向遍历，只有当坐标合法，未经过，字母正确才返回True

代码：

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directions=[(1,0),(0,1),(-1,0),(0,-1)]
        def dfs(i,j,cnt):
            if board[i][j]!=word[cnt]:
                return False
            if cnt==len(word)-1:
                return True
            visited.add((i,j))
            judge=False
            for dx,dy in directions:
                nx,ny=i+dx,j+dy
                if 0<=nx<n and 0<=ny<m and (nx,ny) not in visited:
                    if dfs(nx,ny,cnt+1):
                        judge=True
                        break
            visited.remove((i,j))
            return judge
        
        n=len(board)
        m=len(board[0])
        visited=set()
        for i in range(n):
            for j in range(m):
                if dfs(i,j,0):
                    return True
        return False

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401105123758](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250401105123758.png)



### LC94.二叉树的中序遍历

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

思路：

递归，dfs函数调用先后导致左子树优先加入，然后是根，最后是右子树，即中序遍历

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans=[]
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            ans.append(node.val)
            dfs(node.right)
        
        dfs(root)
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401110611681](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250401110611681.png)



### LC102.二叉树的层序遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

思路：

用双端队列存储每一层的节点，知道上一层的节点数后可以popleft上一层节点，append下一层节点存入level，注意root也可能为None

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    from collections import deque
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        ans=[]
        q=deque()
        q.append(root)
        while q:
            level=[]
            for i in range(len(q)):
                node=q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            ans.append(level)
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401111829109](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250401111829109.png)



### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

思路：

滚动数组背包问题预处理所有子序列回文情况，以左指针作为坐标dfs搜索右指针

代码：

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        ans=[]
        n=len(s)
        dp=[[True]*n for i in range(n)]
        for i in range(n-1,-1,-1):
            for j in range(i+1,n):
                dp[i][j]=(s[i]==s[j]) and dp[i+1][j-1]

        temp=[]
        def dfs(i):
            if i==n:
                ans.append(temp[:])
                return
            for j in range(i,n):
                if dp[i][j]:
                    temp.append(s[i:j+1])
                    dfs(j+1)
                    temp.pop()
                    
        dfs(0)
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401113736307](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250401113736307.png)



### LC146.LRU缓存

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

思路：

建立双向链表，参数除了值还加上在哈希表中的键，LRUCache类中，以双向链表记录历史，初始化容量、哈希表、伪头节点和伪尾节点，定义renew函数将当前访问的已存在节点接在链表最后，get函数利用哈希表找到键对应的值，put函数覆盖已有键对应节点后renew或是初始化节点加在末尾并加入哈希表，若超过容量还需删去头节点。

代码：

```python
class Node:
    def __init__(self,key=0,value=0):
        self.key=key
        self.value=value
        self.prev=None
        self.next=None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity=capacity
        self.hashmap={}
        self.head=Node()
        self.tail=Node()
        self.head.next=self.tail
        self.tail.prev=self.head

    def renew(self,key):
        node=self.hashmap[key]
        node.prev.next=node.next
        node.next.prev=node.prev
        node.prev=self.tail.prev
        node.next=self.tail
        self.tail.prev.next=node
        self.tail.prev=node

    def get(self, key: int) -> int:
        if key in self.hashmap:
            self.renew(key)
            return self.hashmap.get(key).value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.hashmap:
            self.hashmap[key].value=value
            self.renew(key)
        else:
            if len(self.hashmap)==self.capacity:
                self.hashmap.pop(self.head.next.key)
                self.head.next=self.head.next.next
                self.head.next.prev=self.head
            new=Node(key,value)
            self.hashmap[key]=new
            new.prev=self.tail.prev
            new.next=self.tail
            self.tail.prev.next=new
            self.tail.prev=new

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401120140046](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250401120140046.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

基本掌握了链表的OOP写法，dfs递归已经不太熟练了，甚至stack的写法完全忘了（），通过作业题复习了一下dfs递归和背包dp。新学的树感觉大致跟线性结构形式没有很大区别，就是节点加指针，不过出现新的结构新的题目还是得再熟悉一下写法。两门实验课快结课了，下周开始应该能空出很多时间写题。这个月没怎么写作业题之外的练习，感觉月考有点悬。









