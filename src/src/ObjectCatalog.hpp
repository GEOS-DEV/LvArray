/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/**
 * @file   ObjectCatalog.hpp
 * @author Randolph Settgast
 *
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */


#ifndef OBJECTCATALOG_HPP_
#define OBJECTCATALOG_HPP_
#include <unordered_map>
#include <string>
#include <iostream>
#include <memory>
#include "Logger.hpp"
#include "StringUtilities.hpp"

#ifndef OBJECTCATALOGVERBOSE
#define OBJECTCATALOGVERBOSE 0
#endif


#ifndef BASEHOLDSCATALOG
#define BASEHOLDSCATALOG 1
#endif

/**
 * namespace to hold the object catalog classes
 */
namespace cxx_utilities
{


#if ( __cplusplus < 201402L )

#else

#endif

/**
 *  This class provides the base class/interface for the catalog value objects
 *  @tparam BASETYPE This is the base class of the objects that the factory
 * produces.
 *  @tparam ARGS  variadic template pack to hold the parameters needed for the
 * constructor of the BASETYPE
 */
template< typename BASETYPE, typename ... ARGS >
class CatalogInterface
{
public:
  /// This is the type that will be used for the catalog. The catalog is
  // actually instantiated in the BASETYPE
  typedef std::unordered_map< std::string, std::unique_ptr< CatalogInterface< BASETYPE, ARGS... > > > CatalogType;

  /// default constructor.
  CatalogInterface()
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling constructor for CatalogInterface< " << demangle( typeid(BASETYPE).name())
                                                                << " , ... >" );
#endif
  }

  ///default destructor
  virtual ~CatalogInterface()
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling destructor for CatalogInterface< "<< demangle( typeid(BASETYPE).name())
                                                              <<" , ... >" );
#endif
  }

  explicit CatalogInterface( CatalogInterface const & ) = default;
  CatalogInterface( CatalogInterface && ) = default;
  CatalogInterface& operator=( CatalogInterface const & ) = default;
  CatalogInterface& operator=( CatalogInterface && ) = default;

  /**
   * get the catalog from that is stored in the target base class.
   * @return returns the catalog for this
   */
  static CatalogType& GetCatalog()
  {
#if BASEHOLDSCATALOG == 1
    return BASETYPE::GetCatalog();
#else
    static CatalogType catalog;
    return catalog;
#endif
  }

  /**
   * pure virtual to create a new object that derives from BASETYPE
   * @param args these are the arguments to the constructor of the target type
   * @return passes a unique_ptr<BASETYPE> to the newly allocated class.
   */
  virtual std::unique_ptr< BASETYPE > Allocate( ARGS... args ) const = 0;


  static bool hasKeyName( std::string const & objectTypeName )
  {
    return GetCatalog().count( objectTypeName );
  }

  /**
   * static method to create a new object that derives from BASETYPE
   * @param objectTypeName The key to the catalog entry that is able to create
   * the correct type.
   * @param args these are the arguments to the constructor of the target type
   * @return passes a unique_ptr<BASETYPE> to the newly allocated class.
   */
  static std::unique_ptr< BASETYPE > Factory( std::string const & objectTypeName, ARGS... args )
  {
    CatalogInterface< BASETYPE, ARGS... > const * const entry = GetCatalog().at( objectTypeName ).get();
    return entry->Allocate( args ... );
  }

  template< typename TYPE >
  static TYPE& catalog_cast( BASETYPE& object )
  {
    std::string castedName = TYPE::CatalogName();
    std::string objectName = object.getName();

    if( castedName != objectName )
    {
#if OBJECTCATALOGVERBOSE > 1
      GEOS_LOG_RANK( "Invalid Cast of " << objectName << " to " << castedName );
#endif
    }

    return static_cast< TYPE& >(object);
  }

};

/**
 * class to hold allocation capability for specific target derived types
 * @tparam TYPE this is the derived type
 * @tparam BASETYPE this is the base class that TYPE derives from
 * @tparam ARGS constructor arguments
 */
template< typename BASETYPE, typename TYPE, typename ... ARGS >
class CatalogEntry : public CatalogInterface< BASETYPE, ARGS... >
{
public:
  /// default constructor
  CatalogEntry():
    CatalogInterface< BASETYPE, ARGS... >()
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling constructor for CatalogEntry< " << demangle( typeid(TYPE).name())
                                                            << " , " << demangle( typeid(BASETYPE).name())
                                                            << " , ... >" );
#endif
  }

  /// default destructor
  ~CatalogEntry() override final
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling destructor for CatalogEntry< " << demangle( typeid(TYPE).name())
                                                           << " , " << demangle( typeid(BASETYPE).name())
                                                           << " , ... >" );
#endif

  }

  CatalogEntry( CatalogEntry const & source ):
    CatalogInterface< BASETYPE, ARGS... >( source )
  {}

  CatalogEntry( CatalogEntry && source ):
    CatalogInterface< BASETYPE, ARGS... >( std::move( source ))
  {}

  CatalogEntry& operator=( CatalogEntry const & source )
  {
    CatalogInterface< BASETYPE, ARGS... >::operator=( source );
  }

  CatalogEntry& operator=( CatalogEntry && source )
  {
    CatalogInterface< BASETYPE, ARGS... >::operator=( std::move(source));
  }

  /**
   * inherited virtual to create a new object that derives from BASETYPE
   * @param args these are the arguments to the constructor of the target type
   * @return passes a unique_ptr<BASETYPE> to the newly allocated class.
   */
  virtual std::unique_ptr< BASETYPE > Allocate( ARGS... args ) const override final
  {
#if OBJECTCATALOGVERBOSE > 0
    GEOS_LOG_RANK( "Creating type " << demangle( typeid(TYPE).name())
                                    << " from catalog of " << demangle( typeid(BASETYPE).name()));
#endif
#if ( __cplusplus >= 201402L )
    return std::make_unique< TYPE >( args ... );
#else
    return std::unique_ptr< BASETYPE >( new TYPE( args ... ) );
#endif
  }
};


/**
 * a class to generate the catalog entry
 */
template< typename BASETYPE, typename TYPE, typename ... ARGS >
class CatalogEntryConstructor
{
public:
  /**
   * Constructor creates a catalog entry using the key defined by
   * TYPE::CatalogName(), and value of CatalogEntry<TYPE,BASETYPE,ARGS...>.
   * After the constructor is executed, this object may be destroyed without
   * consequence.
   */
  CatalogEntryConstructor()
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling constructor for CatalogueEntryConstructor< " << demangle( typeid(TYPE).name())
                                                                         << " , " << demangle( typeid(BASETYPE).name())
                                                                         << " , ... >" );
#endif

    std::string name = TYPE::CatalogName();
#if ( __cplusplus >= 201402L )
    std::unique_ptr< CatalogEntry< BASETYPE, TYPE, ARGS... > > temp = std::make_unique< CatalogEntry< BASETYPE, TYPE, ARGS... > >();
#else
    std::unique_ptr< CatalogEntry< BASETYPE, TYPE, ARGS... > > temp = std::unique_ptr< CatalogEntry< BASETYPE, TYPE, ARGS... > >( new CatalogEntry< BASETYPE,
                                                                                                                                                    TYPE,
                                                                                                                                                    ARGS... >()  );
#endif
    ( CatalogInterface< BASETYPE, ARGS... >::GetCatalog() ).insert( std::move( std::make_pair( name, std::move( temp ) ) ) );

#if OBJECTCATALOGVERBOSE > 0
    GEOS_LOG_RANK( "Registered " << demangle( typeid(BASETYPE).name())
                                 << " catalogue component of derived type "
                                 << demangle( typeid(TYPE).name())
                                 << " where " << demangle( typeid(TYPE).name())
                                 << "::CatalogueName() = " << TYPE::CatalogName());
#endif
  }

  /// default destuctor
  ~CatalogEntryConstructor()
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling destructor for CatalogueEntryConstructor< " << demangle( typeid(TYPE).name())
                                                                        << " , " << demangle( typeid(BASETYPE).name())
                                                                        << " , ... >" );
#endif
  }

  CatalogEntryConstructor( CatalogEntryConstructor const & ) = delete;
  CatalogEntryConstructor( CatalogEntryConstructor && ) = delete;
  CatalogEntryConstructor& operator=( CatalogEntryConstructor const & ) = delete;
  CatalogEntryConstructor& operator=( CatalogEntryConstructor && ) = delete;

};

/// Specializtion for constructors with empty argument list
template< typename BASETYPE >
class CatalogInterface< BASETYPE >
{
public:
  /// This is the type that will be used for the catalog. The catalog is
  // actually instantiated in the BASETYPE
  typedef std::unordered_map< std::string, std::unique_ptr< CatalogInterface< BASETYPE > > > CatalogType;

  /// default constructor.
  CatalogInterface()
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling constructor for CatalogInterface< " << demangle( typeid(BASETYPE).name())
                                                                << " , ... >" );
#endif
  }

  ///default destructor
  virtual ~CatalogInterface()
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling destructor for CatalogInterface< " << demangle( typeid(BASETYPE).name())
                                                               << " , ... >" );
#endif
  }

  explicit CatalogInterface( CatalogInterface const & ) = default;
  CatalogInterface( CatalogInterface && ) = default;
  CatalogInterface& operator=( CatalogInterface const & ) = default;
  CatalogInterface& operator=( CatalogInterface && ) = default;

  /**
   * get the catalog from that is stored in the target base class.
   * @return returns the catalog for this
   */
  static CatalogType& GetCatalog()
  {
#if BASEHOLDSCATALOG == 1
    return BASETYPE::GetCatalog();
#else
    static CatalogType catalog;
    return catalog;
#endif
  }

  /**
   * pure virtual to create a new object that derives from BASETYPE
   * @param args these are the arguments to the constructor of the target type
   * @return passes a unique_ptr<BASETYPE> to the newly allocated class.
   */
  virtual std::unique_ptr< BASETYPE > Allocate(  ) const = 0;

  /**
   * static method to create a new object that derives from BASETYPE
   * @param objectTypeName The key to the catalog entry that is able to create
   * the correct type.
   * @param args these are the arguments to the constructor of the target type
   * @return passes a unique_ptr<BASETYPE> to the newly allocated class.
   */
  static std::unique_ptr< BASETYPE > Factory( std::string const & objectTypeName )
  {
    CatalogInterface< BASETYPE > const * const entry = GetCatalog().at( objectTypeName ).get();
    return entry->Allocate();
  }

  template< typename TYPE >
  static TYPE& catalog_cast( BASETYPE& object )
  {
    std::string castedName = TYPE::CatalogName();
    std::string objectName = object.getName();

    if( castedName != objectName )
    {
#if OBJECTCATALOGVERBOSE > 1
      GEOS_LOG_RANK( "Invalid Cast of " << objectName << " to " << castedName );
#endif
    }

    return static_cast< TYPE& >(object);
  }

};

template< typename BASETYPE, typename TYPE >
class CatalogEntry< BASETYPE, TYPE > : public CatalogInterface< BASETYPE >
{
public:
  /// default constructor
  CatalogEntry():
    CatalogInterface< BASETYPE >()
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling constructor for CatalogEntry< " << demangle( typeid(TYPE).name())
                                                            << " , " << demangle( typeid(BASETYPE).name())
                                                            << " , ... >" );
#endif
  }

  /// default destructor
  ~CatalogEntry() override final
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling destructor for CatalogEntry< " << demangle( typeid(TYPE).name())
                                                           << " , " << demangle( typeid(BASETYPE).name())
                                                           << " , ... >" );
#endif

  }

  CatalogEntry( CatalogEntry const & source ):
    CatalogInterface< BASETYPE >( source )
  {}

  CatalogEntry( CatalogEntry && source ):
    CatalogInterface< BASETYPE >( std::move( source ))
  {}

  CatalogEntry& operator=( CatalogEntry const & source )
  {
    CatalogInterface< BASETYPE >::operator=( source );
  }

  CatalogEntry& operator=( CatalogEntry && source )
  {
    CatalogInterface< BASETYPE >::operator=( std::move(source));
  }

  virtual std::unique_ptr< BASETYPE > Allocate(  ) const override final
  {
#if OBJECTCATALOGVERBOSE > 0
    GEOS_LOG_RANK( "Creating type " << demangle( typeid(TYPE).name())
                                    << " from catalog of " << demangle( typeid(BASETYPE).name()));
#endif
#if ( __cplusplus >= 201402L )
    return std::make_unique< TYPE >(  );
#else
    return std::unique_ptr< BASETYPE >( new TYPE(  ) );
#endif
  }
};



template< typename BASETYPE, typename TYPE >
class CatalogEntryConstructor< BASETYPE, TYPE >
{
public:
  CatalogEntryConstructor()
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling constructor for CatalogueEntryConstructor< " << demangle( typeid(TYPE).name())
                                                                         << " , " << demangle( typeid(BASETYPE).name())
                                                                         << " , ... >" );
#endif

    std::string name = TYPE::CatalogName();
#if ( __cplusplus >= 201402L )
    std::unique_ptr< CatalogEntry< BASETYPE, TYPE > > temp = std::make_unique< CatalogEntry< BASETYPE, TYPE > >();
#else
    std::unique_ptr< CatalogEntry< BASETYPE, TYPE > > temp = std::unique_ptr< CatalogEntry< BASETYPE, TYPE > >( new CatalogEntry< BASETYPE, TYPE >()  );
#endif
    ( CatalogInterface< BASETYPE >::GetCatalog() ).insert( std::move( std::make_pair( name, std::move( temp ) ) ) );

#if OBJECTCATALOGVERBOSE > 0
    GEOS_LOG_RANK( "Registered " << demangle( typeid(BASETYPE).name())
                                 << " catalogue component of derived type "
                                 << demangle( typeid(TYPE).name())
                                 << " where " << demangle( typeid(TYPE).name())
                                 << "::CatalogueName() = " << TYPE::CatalogName());
#endif
  }

  /// default destuctor
  ~CatalogEntryConstructor()
  {
#if OBJECTCATALOGVERBOSE > 1
    GEOS_LOG_RANK( "Calling destructor for CatalogueEntryConstructor< " << demangle( typeid(TYPE).name())
                                                                        << " , " << demangle( typeid(BASETYPE).name()) << " , ... >" );
#endif
  }

  CatalogEntryConstructor( CatalogEntryConstructor const & ) = delete;
  CatalogEntryConstructor( CatalogEntryConstructor && ) = delete;
  CatalogEntryConstructor& operator=( CatalogEntryConstructor const & ) = delete;
  CatalogEntryConstructor& operator=( CatalogEntryConstructor && ) = delete;

};



}


/**
 * Macro that takes in the base class of the catalog, the derived class, and the
 * argument types for the constructor of
 * the derived class/base class, and create an object of type
 * CatalogEntryConstructor<ClassName,BaseType,__VA_ARGS__> in
 * an anonymous namespace. This should be called from the source file for the
 * derived class, which will result in the
 * generation of a CatalogEntry<BaseType,ClassName,...> prior to main().
 */
#define REGISTER_CATALOG_ENTRY( BaseType, DerivedType, ... ) \
  namespace { cxx_utilities::CatalogEntryConstructor< BaseType, DerivedType, __VA_ARGS__ > catEntry_ ## DerivedType; }

#define REGISTER_CATALOG_ENTRY0( BaseType, DerivedType ) \
  namespace { cxx_utilities::CatalogEntryConstructor< BaseType, DerivedType > catEntry_ ## DerivedType; }

#endif /* OBJECTCATALOG_HPP_ */
