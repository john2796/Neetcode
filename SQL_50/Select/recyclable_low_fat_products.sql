-- https://leetcode.com/problems/recyclable-and-low-fat-products/?envType=study-plan-v2&envId=top-sql-50


Select product_id from Products as p
Where p.low_fats = 'Y' and p.recyclable = 'Y';