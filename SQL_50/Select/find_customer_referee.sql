-- https://leetcode.com/problems/find-customer-referee/description/?envType=study-plan-v2&envId=top-sql-50


Select name from customer as c
Where c.referee_id != '2' or referee_id is null;