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

/* 1211. Queries Quality and Percentage
quality : The average of the ratio between query rating and its position.
poor_query_percentage : The percentage of all queries with rating less than 3.
*/

SELECT 
    query_name,
    ROUND(AVG(rating/position), 2) AS quality,
    ROUND(SUM(CASE WHEN rating < 3 THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS poor_query_percentage
FROM 
    Queries
WHERE query_name is not null
GROUP BY
    query_name;


-- https://leetcode.com/problems/monthly-transactions-i/description/?envType=study-plan-v2&envId=top-sql-50
-- Select the desired columns from the Transactions table
SELECT  
    -- Extract the year and month from the transaction date (first 7 characters: YYYY-MM)
    SUBSTR(trans_date, 1, 7) AS month,
    -- Include the country column
    country,
    -- Count the total number of transactions
    COUNT(id) AS trans_count,
    -- Count the number of approved transactions using a CASE statement
    SUM(CASE WHEN state = 'approved' THEN 1 ELSE 0 END) AS approved_count,
    -- Calculate the total amount of all transactions
    SUM(amount) AS trans_total_amount,
    -- Calculate the total amount of approved transactions using a CASE statement
    SUM(CASE WHEN state = 'approved' THEN amount ELSE 0 END) AS approved_total_amount
-- Specify the table from which to retrieve the data
FROM Transactions
-- Group the results by month and country
GROUP BY month, country;

-- https://leetcode.com/problems/immediate-food-delivery-ii/?envType=study-plan-v2&envId=top-sql-50
Select
    round(avg(order_date = customer_pref_delivery_date) * 100, 2) as immediate_percentage
from Delivery
where (customer_id, order_date) in (
    Select customer_id, min(order_date)
    from Delivery
    group by customer_id
)