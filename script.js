document.addEventListener("DOMContentLoaded", function () {
     const menuToggle = document.querySelector(".menu-toggle");
     const navLinks = document.querySelector(".nav-links");
 
     menuToggle.addEventListener("click", function () {
         console.log("Menu toggle clicked."); // Add this line for testing
         navLinks.classList.toggle("active");
     });
 });
 console.log("Script loaded."); // Add this line at the start of your script
