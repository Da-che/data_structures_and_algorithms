# Assignment #8: 树为主

Updated 1704 GMT+8 Apr 8, 2025

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

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：

已有中序输出，建平衡二叉树，dfs每一步取中间（(lft+rgt+1)//2是左子树可能更长的情况）作为根节点，小的作为左子树，大的作为右子树以满足中序升序。

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def parse(lft,rgt):
            if lft>rgt:
                return None
            mid=(lft+rgt+1)//2
            root=TreeNode(nums[mid])
            root.left=parse(lft,mid-1)
            root.right=parse(mid+1,rgt)
            return root
        return parse(0,len(nums)-1)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250414234756424](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250414234756424.png)



### M27928:遍历树

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：

建多叉树，设parent以便找到根节点，每遍历到一个节点，将子节点排序，先递归处理值小的子节点，输出现节点，再递归值大的子节点

代码：

```python
class TreeNode:
    def __init__(self,val=None):
        self.val = val
        self.children=[]
        self.parent=None


def traverse(node):
    sorted_children=sorted(node.children,key=lambda x:x.val)
    for child in sorted_children:
        if child.val<node.val:
            traverse(child)
    print(node.val)
    for child in sorted_children:
        if child.val>node.val:
            traverse(child)


def main():
    n=int(input())
    Nodes={}
    for i in range(n):
        a=list(map(int,input().split()))
        if a[0] not in Nodes.keys():
            Nodes[a[0]]=TreeNode(a[0])
        current=Nodes[a[0]]
        for j in range(1,len(a)):
            if a[j] not in Nodes.keys():
                Nodes[a[j]]=TreeNode(a[j])
            current.children.append(Nodes[a[j]])
            Nodes[a[j]].parent=current

    while current.parent:
        current=current.parent
    traverse(current)

if __name__ == '__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415101447405](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250415101447405.png)



### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：

dfs

代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        res=0
        def dfs(node,route):
            nonlocal res
            if not node.left and not node.right:
                res+=int(route)
            if node.left:
                dfs(node.left,route+str(node.left.val))
            if node.right:
                dfs(node.right,route+str(node.right.val))
        dfs(root,str(root.val))
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415105155514](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250415105155514.png)



### M22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/22158/

思路：

有前序preorder（根，左子树，右子树）和中序inorder（左子树，根，右子树），返回根preorder[0]，在inorder中找到该值位置root_index，左边为左子树，右边为右子树，知道子树大小，在前序中也可拆分出左右子树，分别递归，左右子树返回的根作为self和right。建好树后后序遍历。

代码：

```python
import sys
class TreeNode:
    def __init__(self,val=None):
        self.val=val
        self.left=None
        self.right=None

def parse(preorder,inorder):
    if not preorder or not inorder:
        return None
    root=TreeNode(preorder[0])
    root.index=inorder.index(preorder[0])
    root.left=parse(preorder[1:root.index+1],inorder[:root.index])
    root.right=parse(preorder[root.index+1:],inorder[root.index+1:])
    return root

def postorder_traversal(node):
    if not node:
        return
    postorder_traversal(node.left)
    postorder_traversal(node.right)
    print(node.val,end='')

def main():
    lines=sys.stdin.readlines()
    for i in range(0,len(lines),2):
        preorder=lines[i].strip()
        inorder=lines[i+1].strip()
        root=parse(preorder,inorder)
        postorder_traversal(root)
        print()

if __name__=='__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415112721448](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250415112721448.png)



### M24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：

用stack建树，先把字母全初始化为节点，用字典存储。把输入压入栈直到右括号，pop当前括号内元素存在children，去掉','即为子节点键，pop掉左括号，栈顶为父节点键。stack[0]为root键。然后正序和反序遍历。

代码：

```python
class TreeNode:
    def __init__(self,val=None):
        self.val=val
        self.children=[]

def parse(bracket):
    stack=[]
    nodes={}
    for i in bracket:
        if i.isalpha():
            nodes[i]=TreeNode(i)
        if i!=')':
            stack.append(i)
        else:
            children=''
            while stack[-1]!='(':
                children+=stack.pop()
            stack.pop()
            children=list(reversed(children.split(',')))
            parent=nodes[stack[-1]]
            for child in children:
                parent.children.append(nodes[child])
    return nodes[stack[0]]

def preorder_traversal(node):
    if not node:
        return
    print(node.val,end='')
    for child in node.children:
        preorder_traversal(child)

def postorder_traversal(node):
    if not node:
        return
    for child in node.children:
        postorder_traversal(child)
    print(node.val,end='')

def main():
    bracket=input()
    root=parse(bracket)
    preorder_traversal(root)
    print()
    postorder_traversal(root)

if __name__=='__main__':
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415130645346](C:\Users\20157\AppData\Roaming\Typora\typora-user-images\image-20250415130645346.png)



### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：

放弃了，写了四五个小时，发现只能满足全正数数组

代码：

```python
WA：
class Node:
    def __init__(self,val=None):
        self.val=val
        self.prev=None
        self.next=None

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        n=len(nums)
        dec=0  #降序数
        current=Node(nums[0])
        q=[]
        lazy={0:1}
        for i in range(1,n):
            new=Node(nums[i])
            lazy[i]=1
            if nums[i]<nums[i-1]:
                dec+=1  #记录降序数
            current.next,new.prev=new,current  #建双向链表
            heapq.heappush(q,(current.val+new.val,i,current))  #最小堆找和最小且最左端的数对
            current=new

        ans=0
        while dec:
            if not q:
                return ans
            sumval,index,current=heapq.heappop(q)  #当前和最小的数对
            if not lazy[index]:
                continue  #排除已被合并的
            if lazy[index]>1:
                lazy[index]-=1
                continue  #排除和已更新的（一定往大了更新）
            ans+=1  #进行合并
            for rgt in range(index+1,n):
                if lazy[rgt]>0:  #找到下一位
                    lazy[rgt]=0
                    break
            if current.val>current.next.val:  #合并前为降序
                dec-=1
            if current.prev:
                if current.val<current.prev.val and sumval>=current.prev.val:  #合并前左为降序，合并后为升序
                    dec-=1
            if current.next.next:
                if current.next.next.val>=current.next.val and current.next.next.val<sumval:  #合并前右为升序，合并后为降序
                    dec+=1

            current.val=sumval  #合并
            if current.next.next:  #更新双向链表，删除右节点
                current.next.next.prev,current.next=current,current.next.next
            else:
                current.next=None  #右侧节点在末尾

            if current.prev:
                i=1
                while not lazy[index-i]:
                    i+=1  #找到左侧节点
                heapq.heappush(q,(current.prev.val+sumval,index-i,current.prev))  #更新左侧优先队列
                lazy[index-i]+=1  #懒删除原队列元素
            if current.next:
                heapq.heappush(q,(current.next.val+sumval,index,current))  #更新右侧优先队列
                lazy[rgt]+=1  #懒删除原队列元素
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这次作业真的好难qwq，不过基本理解树的结构后建树越来越熟练了，三种遍历dfs也基本没问题，OOP写规范还是很赏心悦目的。

最后一题真写不出来了，之后好好研究一下题解吧。

这周期中结束，希望把每日一练尽量做做。太多数据结构有点接受不过来了（emmmm谁说数算简单的），多做练习巩固巩固。









