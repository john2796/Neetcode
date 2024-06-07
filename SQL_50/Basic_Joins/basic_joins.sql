-- # Basic joins

-- LEFT JOIN: keyword returns all record from the left table , and the matching records from the right table The result is 0 records from the right side, if there is no match

-- https://leetcode.com/problems/replace-employee-id-with-the-unique-identifier/?envType=study-plan-v2&envId=top-sql-50
select eu.unique_id as unique_id, e.name as name
from Employees e
left join EmployeeUNI eu
on e.id = eu.id;



