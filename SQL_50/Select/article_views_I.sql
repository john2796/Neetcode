-- https://leetcode.com/problems/article-views-i/description/?envType=study-plan-v2&envId=top-sql-50

-- # Write your MySQL query statement below
Select DISTINCT author_id as id From Views as v
Where v.author_id=v.viewer_id
ORDER BY v.author_id ASC;
