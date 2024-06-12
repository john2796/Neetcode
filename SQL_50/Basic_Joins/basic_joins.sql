-- # Basic joins

-- LEFT JOIN: keyword returns all record from the left table , and the matching records from the right table The result is 0 records from the right side, if there is no match

-- https://leetcode.com/problems/replace-employee-id-with-the-unique-identifier/?envType=study-plan-v2&envId=top-sql-50
select eu.unique_id as unique_id, e.name as name
from Employees e
left join EmployeeUNI eu
on e.id = eu.id;

-- https://leetcode.com/problems/product-sales-analysis-i/description/?envType=study-plan-v2&envId=top-sql-50
-- # Write your MySQL query statement below
-- # show report for product_name, year and price for each sales in the sales table
-- # summary: show product for each sale
select product_name, year, price from Sales s
left join Product p
on s.product_id = p.product_id;

-- https://leetcode.com/problems/customer-who-visited-but-did-not-make-any-transactions/description/?envType=study-plan-v2&envId=top-sql-50
-- - find the IDs of the users who visited without making any transactions 
-- - and the number of times they made these types of visits.
-- - visit_id | customer_ id 
SELECT customer_id,COUNT(customer_id) AS count_no_trans
FROM Visits
LEFT JOIN Transactions ON Visits.visit_id=Transactions.visit_id
WHERE transaction_id is null
GROUP BY customer_id;



-- https://leetcode.com/problems/rising-temperature/description/?envType=study-plan-v2&envId=top-sql-50
/* 
 - find all dates' ID with higher temp 
 - compared to prev dates (yesterday)
 - return result in any order

 get today and yesterday dates and compare them
 use sql functions
 https://www.w3schools.com/sql/func_sqlserver_datediff.asp
*/
SELECT w1.id
FROM Weather w1, Weather w2
WHERE  DATEDIFF(w1.recordDate, w2.recordDate) = 1 -- get diff in 1 day
AND w1.temperature > w2.temperature; -- select w1 that are greater in temp from prev day


-- https://leetcode.com/problems/average-time-of-process-per-machine/description/?envType=study-plan-v2&envId=top-sql-50
/* Average Time of Process per Machine
 - find the average time each machine takes to complete to process
 - time to complete a process is the end_timestamp - start_timestamp
 - average time is total_time / number_of_processes that were run
 - result table should have machine_id , processing_time rounded dto 3 decimal places
*/

select a1.machine_id, round(avg(a2.timestamp - a1.timestamp), 3) as processing_time
from Activity a1
join Activity a2
on a1.machine_id = a2.machine_id and a1.process_id = a2.process_id
and a1.activity_type = 'start' and a2.activity_type = 'end'
group by a1.machine_id;


-- https://leetcode.com/problems/employee-bonus/description/?envType=study-plan-v2&envId=top-sql-50

select e.name, b.bonus from Employee e
left join Bonus b
on b.empId = e.empId 
where b.bonus < 1000 or b.bonus is null;

-- https://leetcode.com/problems/students-and-examinations/description/?envType=study-plan-v2&envId=top-sql-50
-- Select specific columns from the Students, Subjects, and Examinations tables
select 
    s.student_id,           -- Select the student ID from the Students table
    s.student_name,         -- Select the student name from the Students table
    sub.subject_name,       -- Select the subject name from the Subjects table
    count(e.subject_name) as attended_exams -- Count the number of examinations attended by each student for each subject

from 
    Students s              -- From the Students table aliased as 's'
cross join 
    Subjects sub            -- combines each student with each subject, ensuring every student-subject pair is considered.

left outer join 
    Examinations e          -- Left outer join with the Examinations table aliased as 'e', using a left join to ensure all student-subject pairs are retained even if no exams are recorded.
on 
    s.student_id = e.student_id         -- Join condition on student ID
    and sub.subject_name = e.subject_name -- and subject name to match students and their respective subjects in the Examinations table

-- Group the results by student ID, student name, and subject name to aggregate the count of attended exams
group by 
    s.student_id, 
    s.student_name, 
    sub.subject_name 
-- Order the final results by student ID and then by subject name for organized output
order by 
    s.student_id, 
    sub.subject_name;
