-- https://leetcode.com/problems/not-boring-movies/?envType=study-plan-v2&envId=top-sql-50
select * from Cinema
where description != "boring"
and (mod(id, 2)=1) -- odd number
order by rating desc;

-- https://leetcode.com/problems/average-selling-price/?envType=study-plan-v2&envId=top-sql-50
select p.product_id, ifnull(round(sum(units*price)/sum(units),2),0) as average_price 
from Prices p 
left join UnitsSold u
on p.product_id=u.product_id
and u.purchase_date between p.start_date and p.end_date
group by product_id;

-- Write your MySQL query statement below
-- SQL query that reports the average experience years of all the employees for each project, rounded to 2 digits.

-- we want to add experienced_years to Project that's why we select from Project and left join the Employee
SELECT p.project_id, ROUND(AVG(e.experience_years), 2) as average_years
FROM Project p 
LEFT JOIN Employee e
ON p.employee_id = e.employee_id
GROUP BY p.project_id; -- group by will calculate the average automatically

-- Write your MySQL query statement below
-- find the percentage of the users registered in each contest rounded to two decimals.
-- Return the result table ordered by percentage in descending order. In case of a tie, order it by contest_id in ascending order.
select 
contest_id,
round(count(distinct user_id) * 100 / (select count(user_id) from Users), 2) as percentage
from Register
group by contest_id
order by percentage desc, contest_id;