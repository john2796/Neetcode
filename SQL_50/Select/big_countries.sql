-- https://leetcode.com/problems/big-countries/description/?envType=study-plan-v2&envId=top-sql-50

-- # Write your MySQL query statement below
Select name, population, area From World
Where area >= '3000000' or population >= '25000000';