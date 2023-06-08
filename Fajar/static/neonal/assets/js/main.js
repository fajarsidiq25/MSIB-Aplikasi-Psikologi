(function ($) {
    "use strict";    
    
  /*----------------------------------------
  Sticky Header Activation
  ------------------------------------------*/
  $(window).on('scroll', function () {
    if ($(this).scrollTop() > 0) {
      $('.sticky-header').addClass('sticky');
    } else {
      $('.sticky-header').removeClass('sticky');
    }
  });

  /*---------------------------
    Hero Slider Activation
  -----------------------------------*/
  var mySwiper = new Swiper('.slider-container.swiper-container', {
    spaceBetween: 30,
    centeredSlides: true,
    speed: 3000,
    autoplay: false,
    loop: true,
    effect: 'fade',
    fadeEffect: {
      crossFade: true
    },
    /* events */
    on: {
      slideChange: function(){
        var currentSlide = this.realIndex + 1;
        document.querySelector('.current-slide').innerHTML = currentSlide;
      },
      paginationRender: function(){
        var totalSlides = document.getElementsByClassName('swiper-pagination-bullet').length;
        document.querySelector('.total-slides').innerHTML = totalSlides;
      }
    }
  });

  /*---------------------------
    Team Carousel Activation
  -----------------------------------*/
  // Swipper JS
  var mySwiper = new Swiper('.team-carousel .swiper-container', {
    slidesPerView : 1,
    speed: 1000,
    loop: true,
    spaceBetween : 30,
    autoplay: false,
    effect: 'fade',
    fadeEffect: {
      crossFade: true,
    },
    navigation: {
      nextEl: '.team-carousel .swiper-button-next',
      prevEl: '.team-carousel .swiper-button-prev',
    }
  });

  /*---------------------------
    Testimonial Carousel Activation
  -----------------------------------*/
  // Swipper JS
  var mySwiper = new Swiper('.testimonial-carousel.swiper-container', {
    slidesPerView : 1,
    speed: 1000,
    loop: true,
    spaceBetween : 30,
    autoplay: true,
    navigation: {
      nextEl: '.testimonial-carousel .swiper-button-next',
      prevEl: '.testimonial-carousel .swiper-button-prev',
    }
  });

	/*---------------------------
    Magnific Popup
  -----------------------------------*/
  $('.video-popup').magnificPopup({
    type: 'iframe'
	});

	/*-----------------------------------------
    Off Canvas Mobile Menu
  -------------------------------------------*/
  $(".header-action-btn-menu").on('click', function () {
    $("body").addClass('fix');
    $(".mobile-menu-wrapper").addClass('open');
	});

	$(".offcanvas-btn-close,.offcanvas-overlay").on('click', function () {
		$("body").removeClass('fix');
		$(".mobile-menu-wrapper").removeClass('open');
	});
	
	/*----------------------------------------
    Responsive Mobile Menu
  ------------------------------------------*/
  //Variables
  var $offCanvasNav = $('.mobile-menu, .category-menu'),
  $offCanvasNavSubMenu = $offCanvasNav.find('.dropdown');

  //Close Off Canvas Sub Menu
  $offCanvasNavSubMenu.slideUp();

  //Category Sub Menu Toggle
  $offCanvasNav.on('click', 'li a, li .menu-expand', function(e) {
  var $this = $(this);
    if ( ($this.parent().attr('class').match(/\b(menu-item-has-children|has-children|has-sub-menu)\b/)) && ($this.attr('href') === '#' || $this.hasClass('menu-expand')) ) {
      e.preventDefault();
      if ($this.siblings('ul:visible').length){
        $this.parent('li').removeClass('active');
        $this.siblings('ul').slideUp();
      } else {
        $this.parent('li').addClass('active');
        $this.closest('li').siblings('li').removeClass('active').find('li').removeClass('active');
        $this.closest('li').siblings('li').find('ul:visible').slideUp();
        $this.siblings('ul').slideDown();
      }
    }
  });

  /*----------------------------------------*/
  /*  When document is loading, do
  /*----------------------------------------*/
  var varWindow = $(window);
  varWindow.on('load', function() {
    AOS.init({
      once: true,
    });
  });

  /*----------------------------------------*/
  /*  Splitting When Document is Loading, do
  /*----------------------------------------*/
  Splitting();

  /*----------------------------------------*/
  /*  Datepicker
  /*----------------------------------------*/
  $( ".date-pic" ).datepicker({
    showOn:"button",
    buttonText: '<i class="icofont-calendar"></i>'
  });

  /*----------------------------------------*/
  /*  Project Filter
  /*----------------------------------------*/
  $('.gallery-filter-nav').on( 'click', 'button', function() {
      var filterValue = $(this).attr('data-filter');
      $grid.isotope({ filter: filterValue });
      
      $(this).siblings('.active').removeClass('active');
      $(this).addClass('active');
  });

  var $grid = $('.gallery-filter-wrapper').isotope({
      itemSelector: '.filter-item',
      percentPosition: true,
  });

  /*----------------------------------------*/
  /*  Lightgallery Active
  /*----------------------------------------*/
  $(".popup-gallery").magnificPopup({
    delegate: 'a',
    type: 'image'
  });
	/*-------------------------
     Ajax Contact Form 
  ---------------------------*/
  $(function() {

    // Get the form.
    var form = $('#contact-form');

    // Get the messages div.
    var formMessages = $('.form-messege');

    // Set up an event listener for the contact form.
    $(form).on('submit', function(e) {
        // Stop the browser from submitting the form.
        e.preventDefault();

        // Serialize the form data.
        var formData = $(form).serialize();

        // Submit the form using AJAX.
        $.ajax({
          type: 'POST',
          url: $(form).attr('action'),
          data: formData
        })
        .done(function(response) {
          // Make sure that the formMessages div has the 'success' class.
          $(formMessages).removeClass('error');
          $(formMessages).addClass('success');

          // Set the message text.
          $(formMessages).text(response);

          // Clear the form.
          $('#contact-form input,#contact-form textarea').val('');
        })
        .fail(function(data) {
          // Make sure that the formMessages div has the 'error' class.
          $(formMessages).removeClass('success');
          $(formMessages).addClass('error');

          // Set the message text.
          if (data.responseText !== '') {
              $(formMessages).text(data.responseText);
          } else {
              $(formMessages).text('Oops! An error occured and your message could not be sent.');
          }
        });
    });

  });
  /*----------------------------------------*/
  /*  Scroll to top
  /*----------------------------------------*/
  function scrollToTop() {
    var $scrollUp = $('#scroll-top'),
      $lastScrollTop = 0,
      $window = $(window);
    $window.on('scroll', function () {
      var st = $(this).scrollTop();
      if (st > $lastScrollTop) {
          $scrollUp.removeClass('show');
      } else {
        if ($window.scrollTop() > 200) {
            $scrollUp.addClass('show');
        } else {
            $scrollUp.removeClass('show');
        }
      }
      $lastScrollTop = st;
    });

    $scrollUp.on('click', function (evt) {
      $('html, body').animate({scrollTop: 0}, 600);
      evt.preventDefault();
    });
  }
  scrollToTop();

})(jQuery);

