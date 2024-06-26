class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        // base case
        if (root == NULL){
            return NULL;
        }
        // call left and right subtree
        invertTree(root->left);
        invertTree(root->right);

        // swap the nodes
        TreeNode* temp = root->right;
        root->right = root->left;
        root->left = temp;

        return root;
    }
};

class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        int max_left = maxDepth(root->left);
        int max_right = maxDepth(root->right);
        return max(max_left, max_right) + 1;
    }
};

// 543. Diameter of Binary Tree
class Solution {
  public:
    int diameterOfBinaryTree(TreeNode* root){
    int result = 0;
    dfs(root, result);
    return result;
  }

  private:
    int dfs(TreeNode* root, int& result) {
    if (root == NULL) return 0;

    int left = dfs(root->left, result);
    int right = dfs(root->right, result);
    result = max(result, left+right);
    return 1 + max(left, right);
  }
};
