-- https://leetcode.com/problems/article-views-i/description/?envType=study-plan-v2&envId=top-sql-50
Select DISTINCT author_id as id From Views as v
Where v.author_id=v.viewer_id
ORDER BY v.author_id ASC;

-- https://leetcode.com/problems/big-countries/description/?envType=study-plan-v2&envId=top-sql-50
Select name, population, area From World
Where area >= '3000000' or population >= '25000000';

-- https://leetcode.com/problems/find-customer-referee/description/?envType=study-plan-v2&envId=top-sql-50
Select name from customer as c
Where c.referee_id != '2' or referee_id is null;

-- https://leetcode.com/problems/invalid-tweets/description/?envType=study-plan-v2&envId=top-sql-50
select tweet_id from tweets
where CHAR_LENGTH(content) > 15;

-- https://leetcode.com/problems/recyclable-and-low-fat-products/?envType=study-plan-v2&envId=top-sql-50
Select product_id from Products as p
Where p.low_fats = 'Y' and p.recyclable = 'Y';
  