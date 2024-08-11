-- Lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
-- INNER JOIN WITH GROUP BY
SELECT genre AS genre, COUNT(*) AS number_of_shows
FROM shows
GROUP BY genre
ORDER BY number_of_shows DESC;
